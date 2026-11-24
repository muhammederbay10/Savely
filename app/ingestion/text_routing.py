"""Decides whether a text sent right after a share names that post's collection (record 0018).

Ingestion binds a text to the user's last share sent before it, within
`COLLECTION_HINT_WINDOW_SECONDS`, and leaves its route `UNDECIDED`
(`app.ingestion.service`). The chat worker settles it here, before the
response agent runs:

1. Under the user's row lock, the receipt is read again: a route an earlier
   try decided is followed, never asked again.
2. An item that has a collection by now (an earlier text named it) makes the
   text chat, with no model call.
3. Otherwise the decision model answers one yes/no question about the text
   (`ROUTING_PROMPTS`): does it only name a collection for that post? Yes at
   or above `COLLECTION_HINT_MIN_PROBABILITY` makes it a collection hint.
   Below it, an unanswered question, a failed call or a call that took longer
   than `COLLECTION_HINT_DECISION_TIMEOUT_SECONDS` (provider retries included)
   makes it chat, and that is kept: a missed label costs little, a swallowed
   question looks broken. The timeout is at most half the chat run's
   deadline, so the rest of the run is left to answer.
4. Under the user's row lock again, the decision is kept in the receipt and,
   for a hint, the item gets its collection, in one go. Whether the item's
   save confirmation will miss it is read under the same lock and kept too
   (`confirm_collection`): the confirmation had already started, or the
   item's indexing failed for good, so no confirmation will come. Only then
   does the chat worker send one short collection confirmation, and on every
   try it decides the same. An item whose indexing fails for good after the
   label was set gets that confirmation from the indexer instead
   (`app.pipeline.recovery.acknowledgment`).

The receipt in Redis is written before the database commits. If the commit
fails, the label is set again under the lock before a hint is followed, by a
later try or by an overlapping one that finds the receipt decided
(`set_collection_if_missing` changes nothing when it is already set), so a
hint is never confirmed without its label.

Holding the user's row lock keeps all of this from racing "delete my data",
which takes the same lock; a user deleted meanwhile has no receipt left, and
nothing is written. The model is never called while the lock is held.
"""

import asyncio
import logging
import uuid
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import ChatMessage
from app.ingestion.prompts import RoutingPrompts
from app.ingestion.repository import is_unlabelled_item, lock_user, set_collection_if_missing
from app.ingestion.session_state import RedisIngestionState, TextReceipt, TextRoute
from app.models.base import Classifier, YesNoAnswer, YesNoQuestion
from app.models.errors import ModelError
from app.pipeline.recovery.runs import save_confirmation_misses_collection

logger = logging.getLogger(__name__)

COLLECTION_NAME_QUESTION_KEY: Final = "names_collection"


async def collection_name_probability(
    classifier: Classifier, prompts: RoutingPrompts, text: str
) -> float | None:
    """Ask the decision model how likely a text only names a collection for the post before it.

    Args:
        classifier: The decision model.
        prompts: The wording of the question.
        text: The user's text, as they wrote it.

    Returns:
        The probability of yes, from 0 to 1; `None` when the model left the
        question unanswered.

    Raises:
        ModelError: If the call failed.
    """
    result = await classifier.decide(
        [prompts.collection_name_input.format(text=text)],
        [
            YesNoQuestion(
                key=COLLECTION_NAME_QUESTION_KEY, instructions=prompts.collection_name_question
            )
        ],
    )
    answer = result.answers.get(COLLECTION_NAME_QUESTION_KEY)
    return answer.probability_yes if isinstance(answer, YesNoAnswer) else None


class TextRouter:
    """Settles the route of a text the chat worker is about to handle."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        state: RedisIngestionState,
        classifier: Classifier,
        prompts: RoutingPrompts,
        *,
        min_probability: float,
        decision_timeout_seconds: float,
    ) -> None:
        """Create the router.

        Args:
            sessions: Opens a database session.
            state: Ingestion's per-user state, which holds the receipts.
            classifier: The decision model.
            prompts: The wording of the question.
            min_probability: `COLLECTION_HINT_MIN_PROBABILITY`.
            decision_timeout_seconds: `COLLECTION_HINT_DECISION_TIMEOUT_SECONDS`:
                how long the whole decision call may take, retries included.
        """
        self._sessions = sessions
        self._state = state
        self._classifier = classifier
        self._prompts = prompts
        self._min_probability = min_probability
        self._decision_timeout_seconds = decision_timeout_seconds

    async def settle(
        self, user_id: uuid.UUID, message: ChatMessage, receipt: TextReceipt
    ) -> TextReceipt | None:
        """Return the text's receipt with its route decided, deciding it once if needed.

        Args:
            user_id: The sender.
            message: The text.
            receipt: Its receipt, as the job read it.

        Returns:
            The receipt, routed `CHAT` or `COLLECTION_HINT`; `None` if the user
            or the receipt is gone.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database failed.
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        if receipt.route is TextRoute.CHAT:
            return receipt
        if receipt.route is TextRoute.COLLECTION_HINT:
            return await self._apply_again(user_id, message)
        names_collection = await self._names_collection(user_id, message, receipt)
        return await self._keep_decision(user_id, message, names_collection=names_collection)

    async def _names_collection(
        self, user_id: uuid.UUID, message: ChatMessage, receipt: TextReceipt
    ) -> bool:
        if receipt.saved_item_id is None or not await self._is_unlabelled(
            user_id, receipt.saved_item_id
        ):
            self._log(user_id, TextRoute.CHAT, "item_labelled", probability=None)
            return False
        deadline = asyncio.timeout(self._decision_timeout_seconds)
        try:
            async with deadline:
                probability = await collection_name_probability(
                    self._classifier, self._prompts, message.text
                )
        except TimeoutError:
            if not deadline.expired():
                # The chat run's own deadline: the job tries again.
                raise
            logger.warning("text_route_decision_timed_out", extra={"user_id": str(user_id)})
            self._log(user_id, TextRoute.CHAT, "decision_timed_out", probability=None)
            return False
        except ModelError as error:
            logger.warning(
                "text_route_decision_failed",
                extra={"user_id": str(user_id), "error": type(error).__name__},
            )
            self._log(user_id, TextRoute.CHAT, "decision_failed", probability=None)
            return False
        if probability is None:
            self._log(user_id, TextRoute.CHAT, "unanswered", probability=None)
            return False
        names_collection = probability >= self._min_probability
        route = TextRoute.COLLECTION_HINT if names_collection else TextRoute.CHAT
        reason = "above_threshold" if names_collection else "below_threshold"
        self._log(user_id, route, reason, probability=probability)
        return names_collection

    async def _is_unlabelled(self, user_id: uuid.UUID, saved_item_id: uuid.UUID) -> bool:
        async with self._sessions() as session:
            return await is_unlabelled_item(session, user_id, saved_item_id)

    async def _keep_decision(
        self, user_id: uuid.UUID, message: ChatMessage, *, names_collection: bool
    ) -> TextReceipt | None:
        async with self._sessions.begin() as session:
            current = await self._locked_receipt(session, user_id, message)
            if current is None or current.route is TextRoute.CHAT:
                return current
            if current.route is TextRoute.COLLECTION_HINT:
                # An overlapping try decided first: its decision stands, but its
                # database write may have rolled back.
                return await self._label_again(session, user_id, message, current)
            decided = current.model_copy(update={"route": TextRoute.CHAT})
            if names_collection:
                hint = await self._label(session, user_id, message, current)
                decided = decided if hint is None else hint
            await self._state.save_text_receipt(
                user_id, message.platform, message.message_id, decided
            )
            return decided

    async def _apply_again(self, user_id: uuid.UUID, message: ChatMessage) -> TextReceipt | None:
        async with self._sessions.begin() as session:
            current = await self._locked_receipt(session, user_id, message)
            if current is None or current.route is not TextRoute.COLLECTION_HINT:
                return current
            return await self._label_again(session, user_id, message, current)

    async def _label_again(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        message: ChatMessage,
        receipt: TextReceipt,
    ) -> TextReceipt:
        """Set a decided hint's collection again, in case the try that decided it didn't commit.

        Runs under the user's row lock, and commits before the hint is followed.
        """
        relabelled = await self._label(session, user_id, message, receipt)
        if relabelled is None:
            return receipt
        await self._state.save_text_receipt(
            user_id, message.platform, message.message_id, relabelled
        )
        return relabelled

    async def _label(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        message: ChatMessage,
        receipt: TextReceipt,
    ) -> TextReceipt | None:
        """Give the bound item the text as its collection; `None` if it already has one."""
        if receipt.saved_item_id is None:
            return None
        if not await set_collection_if_missing(
            session, user_id, receipt.saved_item_id, message.text
        ):
            return None
        # Under the row lock the indexer also takes to start the save
        # confirmation and to fail the item.
        missed = await save_confirmation_misses_collection(session, receipt.saved_item_id)
        return receipt.model_copy(
            update={"route": TextRoute.COLLECTION_HINT, "confirm_collection": missed}
        )

    async def _locked_receipt(
        self, session: AsyncSession, user_id: uuid.UUID, message: ChatMessage
    ) -> TextReceipt | None:
        if not await lock_user(session, user_id):
            return None
        return await self._state.text_receipt(user_id, message.platform, message.message_id)

    def _log(
        self, user_id: uuid.UUID, route: TextRoute, reason: str, *, probability: float | None
    ) -> None:
        logger.info(
            "text_route_decided",
            extra={
                "user_id": str(user_id),
                "route": route,
                "reason": reason,
                "probability": probability,
                "prompts_version": self._prompts.version,
            },
        )
