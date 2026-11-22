"""Sends an agent answer, and only then makes a deletion question confirmable.

A "delete my data" question must reach the user before their next message may
confirm it, so its request is stored only after every message of the answer
was sent. A question that failed to send, or was skipped because the reply
window had closed, leaves nothing that can be confirmed.

The request is planned before sending (`deletion_request`): it waits for the
message after the user's last one, and its deadline counts from then. The chat
worker (`app/agent/chat.py`, record 0015 Q5) keeps the planned request with the
rest of the answer's exchange and stores it after sending, trying again after a
failure, so a Redis outage right after sending can't lose it, and storing it
late never lets a later message confirm or gives more time. `send_answer` does
both steps at once, for callers with nothing else to store.
"""

import uuid
from datetime import datetime

from app.agent.agent import AgentAnswer
from app.agent.deletion_requests import PendingDeletion
from app.agent.replies.answers import AnswerSender, DeliveryReport
from app.agent.tools import Toolbox


def deletion_request(
    toolbox: Toolbox, answer: AgentAnswer, last_message_number: int
) -> PendingDeletion | None:
    """Plan the request an answer's deletion question makes confirmable once delivered.

    Args:
        toolbox: Plans the request with its deadline.
        answer: The agent's answer.
        last_message_number: The number of the user's last message, read before
            sending; the question waits for the one after it.

    Returns:
        The request to store once the whole answer was sent; `None` if the
        answer asks no deletion question.
    """
    if answer.deletion_confirmation is None:
        return None
    return toolbox.deletion_request(answer.deletion_confirmation, last_message_number)


async def send_answer(
    sender: AnswerSender,
    toolbox: Toolbox,
    user_id: uuid.UUID,
    recipient_id: str,
    answer: AgentAnswer,
    last_user_message_at: datetime,
    last_message_number: int,
) -> DeliveryReport:
    """Send an answer; store a deletion question's request once all of it was sent.

    Args:
        sender: Sends the intro and the results.
        toolbox: Plans and stores a delivered deletion question's request.
        user_id: The DM's sender.
        recipient_id: Their platform id. After a deletion, take it from
            `answer.deleted_recipients`, read before the user's rows were deleted.
        answer: The agent's answer.
        last_user_message_at: When the user last wrote, as the platform reported it.
        last_message_number: The number of the user's last message, read
            before sending; a deletion question waits for the one after it.

    Returns:
        What was sent.

    Raises:
        MessengerError: If a message could not be sent; nothing is stored then.
        redis.exceptions.RedisError: If the delivered question's request could not be stored.
    """
    request = deletion_request(toolbox, answer, last_message_number)
    report = await sender.send(recipient_id, answer.messages, last_user_message_at)
    if request is not None and report.complete:
        await toolbox.keep_deletion_request(user_id, request)
    return report
