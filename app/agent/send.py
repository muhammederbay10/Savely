"""Sends an agent answer, and only then makes a deletion question confirmable.

The chat worker (`app/agent/chat.py`, record 0015 Q5) calls `send_answer` for every
answer. A "delete my data" question must reach the user before their next
message may confirm it, so the request is stored only after every message of
the answer was sent. A question that failed to send, or was skipped because the
reply window had closed, leaves nothing that can be confirmed.
"""

import uuid
from datetime import datetime

from app.agent.agent import AgentAnswer
from app.agent.replies.answers import AnswerSender, DeliveryReport
from app.agent.tools import Toolbox


async def send_answer(
    sender: AnswerSender,
    toolbox: Toolbox,
    user_id: uuid.UUID,
    recipient_id: str,
    answer: AgentAnswer,
    last_user_message_at: datetime,
    last_message_number: int,
) -> DeliveryReport:
    """Send an answer; record a deletion question as delivered once all of it was sent.

    Args:
        sender: Sends the intro and the results.
        toolbox: Records a delivered deletion question.
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
        MessengerError: If a message could not be sent; nothing is recorded then.
        redis.exceptions.RedisError: If the delivered question could not be recorded.
    """
    report = await sender.send(recipient_id, answer.messages, last_user_message_at)
    if answer.deletion_confirmation is not None and report.complete:
        await toolbox.confirmation_delivered(
            user_id, answer.deletion_confirmation, last_message_number
        )
    return report
