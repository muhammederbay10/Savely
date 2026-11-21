"""Builds the chat worker's responder on fake models, a mock Send API, Postgres and Redis.

A conversation runs the way production runs it: ingestion stores the text and
queues its job, and the responder answers it from ingestion's payload, with the
agent's real tools, short-term memory in Redis and the profile in Postgres.
Only the text model (scripted) and Meta's Send API (recorded) are fakes. Every
id, name and text is synthetic.
"""

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from arq.connections import ArqRedis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import ChatMessage
from app.agent.chat import ChatPayloads, ChatResponder
from app.agent.deletion_requests import RedisDeletionRequests
from app.agent.replies.answers import AnswerSender
from app.db.deletion import UserDataOutsidePostgres
from app.db.models import (
    ConversationMessage,
    Entity,
    ItemEntity,
    ItemStatus,
    SavedItem,
    User,
)
from app.ingestion.erasure import IngestionDataEraser
from app.ingestion.jobs import CHAT_QUEUE, ArqJobQueue
from app.ingestion.service import IngestionService
from app.ingestion.session_state import RedisIngestionState
from app.memory.conversation import ConversationStore
from app.memory.session import SessionMemory
from app.models.base import GenerationResult
from app.models.errors import ModelError
from app.pipeline.media.storage import LocalMediaStore
from tests.agent.agent_support import AgentUnderTest, FakeClock, build_agent
from tests.agent.retrieval_support import BEBEK, place_fields
from tests.agent.support import NOW as MESSENGER_NOW
from tests.agent.support import RecordingSendApi, build_messenger, sent
from tests.db_factories import INSTAGRAM

HINT_WINDOW = timedelta(seconds=30)
SESSION_LIFETIME = timedelta(hours=24)
HISTORY_MESSAGES = 20
RUN_DEADLINE_SECONDS = 10.0
# Plenty of successful Send API answers for one test's conversation.
SEND_ANSWERS = 40
# Texts are sent within Instagram's 24-hour window of the mock Send API's clock.
FIRST_TEXT_AT = MESSENGER_NOW - timedelta(hours=1)


@dataclass
class ChatStores:
    """Every store a conversation touches, shared by the turns of one test."""

    sessions: async_sessionmaker[AsyncSession]
    redis: ArqRedis
    state: RedisIngestionState
    jobs: ArqJobQueue
    requests: RedisDeletionRequests
    memory: SessionMemory
    media: LocalMediaStore
    clock: FakeClock
    ingestion: IngestionService
    texts_sent: int = 0
    users: list[uuid.UUID] = field(default_factory=list)

    def outside_postgres(self) -> list[UserDataOutsidePostgres]:
        """The stores the chat worker's eraser erases, as `app.agent.worker` builds it."""
        return [
            IngestionDataEraser(self.state, self.jobs),
            self.requests,
            self.memory,
            self.media,
        ]

    async def new_user(self, sender_id: str) -> uuid.UUID:
        """A user on first contact, with their Instagram account."""
        user_id = await self.ingestion.resolve_user(INSTAGRAM, sender_id)
        self.users.append(user_id)
        return user_id

    async def send_text(
        self, user_id: uuid.UUID, text: str, *, quick_reply_payload: str | None = None
    ) -> uuid.UUID:
        """The user writes to the bot; ingestion queues the chat job. Returns its receipt id."""
        message_id = f"mid-synthetic-{uuid.uuid4().hex}"
        await self.ingestion.ingest_chat_message(
            ChatMessage(
                user_id=user_id,
                platform=INSTAGRAM,
                text=text,
                message_id=message_id,
                received_at=FIRST_TEXT_AT + timedelta(minutes=self.texts_sent),
                quick_reply_payload=quick_reply_payload,
            )
        )
        self.texts_sent += 1
        receipt = await self.state.text_receipt(user_id, INSTAGRAM, message_id)
        assert receipt is not None
        return receipt.id

    async def clean_up(self) -> None:
        for user_id in self.users:
            await self.state.erase_user(user_id)
            await self.requests.erase_user(user_id, ())
            await self.memory.erase_user(user_id, ())
        users = {str(user_id) for user_id in self.users}
        queued = await self.redis.queued_jobs(queue_name=CHAT_QUEUE)
        await self.jobs.discard([job.job_id for job in queued if job.kwargs["user_id"] in users])


@pytest.fixture
async def chat_stores(
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_redis: ArqRedis,
    ingestion_state: RedisIngestionState,
    tmp_path: Path,
) -> AsyncIterator[ChatStores]:
    jobs = ArqJobQueue(ingestion_redis)
    stores = ChatStores(
        sessions=session_factory,
        redis=ingestion_redis,
        state=ingestion_state,
        jobs=jobs,
        requests=RedisDeletionRequests(ingestion_redis),
        memory=SessionMemory(
            ingestion_redis, lifetime=SESSION_LIFETIME, max_messages=HISTORY_MESSAGES
        ),
        media=LocalMediaStore(tmp_path / "media"),
        clock=FakeClock(),
        ingestion=IngestionService(session_factory, jobs, ingestion_state, HINT_WINDOW),
    )
    try:
        yield stores
    finally:
        await stores.clean_up()


@dataclass
class ChatUnderTest:
    responder: ChatResponder
    agent: AgentUnderTest
    send_api: RecordingSendApi

    def sent_texts(self) -> list[str]:
        return [body["message"]["text"] for body in self.sent_bodies()]

    def sent_bodies(self) -> list[dict[str, dict[str, object]]]:
        return [json.loads(request.content) for request in self.send_api.requests]


def build_chat(
    stores: ChatStores,
    script: Sequence[GenerationResult | ModelError],
    *,
    send_answers: Sequence[httpx.Response | Exception] | None = None,
    **agent_settings: int,
) -> ChatUnderTest:
    """The responder with the real tools and stores, a scripted text model and a mock Send API."""
    agent = build_agent(
        stores.sessions,
        script,
        deletion_requests=stores.requests,
        outside_postgres=stores.outside_postgres(),
        clock=stores.clock,
        **agent_settings,
    )
    messenger = build_messenger(send_answers or [sent() for _ in range(SEND_ANSWERS)])
    responder = ChatResponder(
        payloads=ChatPayloads(stores.redis),
        memory=stores.memory,
        conversations=ConversationStore(stores.sessions),
        agent=agent.agent,
        toolbox=agent.toolbox,
        sender=AnswerSender(messenger.messenger),
        run_deadline_seconds=RUN_DEADLINE_SECONDS,
        now=stores.clock,
    )
    return ChatUnderTest(responder=responder, agent=agent, send_api=messenger.api)


async def seed_place(
    stores: ChatStores,
    user_id: uuid.UUID,
    name: str,
    *,
    district: str,
    price: str,
    saved_days_ago: int,
) -> SavedItem:
    """A saved restaurant post linked to its place, with a price on the place."""
    async with stores.sessions() as session, session.begin():
        entity = Entity(
            type="place",
            source="google_places",
            external_id=f"synthetic-{uuid.uuid4().hex}",
            name=name,
            location=BEBEK,
            country="Türkiye",
            city="İstanbul",
            district=district,
            data={"price": price, "link": f"https://maps.example.test/{uuid.uuid4().hex}"},
        )
        item = SavedItem(
            user_id=user_id,
            platform=INSTAGRAM,
            media_type="reel",
            source_message_id=f"mid-synthetic-save-{uuid.uuid4().hex}",
            type="place",
            category="restaurant",
            collection="Food",
            status=ItemStatus.DONE,
            fields=place_fields((name, district, None)),
            saved_at=MESSENGER_NOW - timedelta(days=saved_days_ago),
        )
        session.add_all([entity, item])
        await session.flush()
        session.add(ItemEntity(saved_item_id=item.id, entity_id=entity.id))
    return item


async def stored_messages(stores: ChatStores, user_id: uuid.UUID) -> list[tuple[str, str]]:
    """The user's rows in `conversation_msg`, oldest first, as (role, content)."""
    async with stores.sessions() as session:
        rows = await session.execute(
            select(ConversationMessage.role, ConversationMessage.content)
            .where(ConversationMessage.user_id == user_id)
            .order_by(ConversationMessage.at)
        )
        return [(role, content) for role, content in rows]


async def count_messages(stores: ChatStores, user_id: uuid.UUID) -> int:
    async with stores.sessions() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(ConversationMessage)
            .where(ConversationMessage.user_id == user_id)
        )
    return count or 0


async def stored_user(stores: ChatStores, user_id: uuid.UUID) -> User | None:
    async with stores.sessions() as session:
        return await session.get(User, user_id)
