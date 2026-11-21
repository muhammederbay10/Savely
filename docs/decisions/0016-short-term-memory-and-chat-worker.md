# 0016: Short-term memory, the basic profile and the chat worker

- Status: proposed with SAV-18; it takes effect when SAV-18's pull request is merged. The open questions below are for Muhammed.
- Date: 2026-11-21
- Ticket: SAV-18

## Context

ARCHITECTURE §3.7 says short-term memory holds the last ~20 messages and the **working set** (the last numbered results and the filters that produced them) in Redis for about 24 hours, so "the second one", "show more" and "only the cheap ones" refine the last search. SAV-18 also adds a basic profile (home area, language) and stores messages in `conversation_msg`.

The owner update of 20 Nov 2026 on SAV-18 (record 0015, Q2, Q5 and Q9) moved the chat worker here: ingestion already queues `answer_chat_message` jobs, but nothing ran them. Record 0015 fixed the agent's contract: the working set is passed in (`AgentTurn.working_set`) and returned (`AgentAnswer.working_set`); the worker sets `deletion_confirmed` only for the "Yes, delete" payload; every new store joins `UserDataEraser`.

## Decision

1. **Three Redis keys per user** (`app/memory/session.py`), `savely:memory:<user id>:history|working_set|replies`. Every write restarts the lifetime of all three (`SESSION_TTL_SECONDS`, 86400 in `.env.example`), so a session ends a day after the last exchange. The history is a list appended and trimmed to `SESSION_HISTORY_MESSAGES` (20) in one transaction, so two answers that end together keep both exchanges; the working set is replaced whole.
2. **The working set carries the searches.** `AgentAnswer.working_set` stays the ordered ids. Two additive fields carry the rest of §3.7's "filters that produced it": the tools report each search's tool and checked arguments (`ToolOutcome.search`, `SearchUsed`), the agent returns them as `AgentAnswer.searches`, and the worker passes them back with how many results were shown (`AgentTurn.working_set_searches`, `AgentTurn.working_set_shown`). The model sees the numbered list, which results were not shown yet, and the searches; to narrow, it repeats the search with one more filter, so "only the cheap ones" gets the same filters plus `price`. A turn that lists items without searching ("show more") keeps the earlier searches. An answer that lists no items keeps the last working set, so "thanks" doesn't lose it.
3. **The basic profile is the `user` row.** A new tool, `set_profile(home_area?, country?)`, keeps where the user lives and their ISO country code when they tell it; a country given mid-turn is used by the rest of that turn. The home area is shown to the model on every turn ("near home" needs no question). The model asks once when a question needs a value it doesn't know. `user.language` is the language of the user's last message (the answer's BCP 47 tag), stored by the worker; the save confirmations already read it.
4. **The chat worker** (`app/agent/worker.py`, `python -m app.agent.worker`, Compose service `chat_worker`) reads only the chat queue. One job (`app/agent/chat.py`): read the payload, load profile and memory, run the agent for the sender only, send with `send_answer` within the 24-hour window counted from the message's `received_at`, then store the exchange in Redis, `conversation_msg` and `user.language`. Nothing is stored after a deletion.
5. **Retries like the indexing worker**, before sending only: `CHAT_RUN_DEADLINE_SECONDS` bounds the preparation; a model outage, the deadline, Redis or Postgres raise `arq.Retry` after `CHAT_RETRY_DELAY_SECONDS` × the try, up to `CHAT_MAX_TRIES`, and the last try sends the fixed error message in the user's language (English if it has no wording). A failure that would repeat sends the error message at once. Sending is never retried: right before it, the job sets a mark in `replies`, and a later try that finds it sends nothing (record 0011's rule that a possibly delivered message is not sent twice).
6. **"Yes, delete"**: `deletion_confirmed = confirms_deletion(message.quick_reply_payload)`. The tools' existing checks make it count only on the sender's next message after a delivered question; a typed "yes", a later tap and another user's tap delete nothing (tests in `tests/agent/test_chat_worker.py`).
7. **"Delete my data"** reaches the new stores: `build_toolbox` gives `UserDataEraser` ingestion's state and jobs, the pending request, `SessionMemory` and the media store. `conversation_msg` rows and the profile go with the `user` row (existing cascade). After the erasure the worker writes nothing for the user, not even the reply mark.

## Alternatives considered

- **One JSON value per user for the whole session**: simpler, but two answers ending together would lose one exchange.
- **Re-running the last search in code for a follow-up**: the model would no longer decide what "only the cheap ones" means; repeating the search with the arguments shown keeps the decision with the model and the checks with the tools.
- **A per-user lock around a whole turn**: would order concurrent messages, but the agent's tools take the user's row lock themselves; the list append and the reply mark make concurrent turns safe without it.
- **Reading the history from `conversation_msg`**: would not expire with the reply window, and costs a query per turn; Postgres keeps the record, Redis serves the turn.

## Consequences

- **Prompt version `2026-11-21.1`**: adds the profile line, the follow-up rules, `set_profile` and the working set's searches to `2026-11-20.3`. Like it, it is scored on the evaluation set when SAV-19/20 exist; there is no agent eval set yet, so there are no before/after numbers.
- **New settings:** `SESSION_TTL_SECONDS`, `SESSION_HISTORY_MESSAGES`, `CHAT_CONCURRENCY`, `CHAT_MAX_TRIES`, `CHAT_RETRY_DELAY_SECONDS`, `CHAT_RUN_DEADLINE_SECONDS`, `CHAT_JOB_TIMEOUT_SECONDS` (`ChatSettings`).
- **No schema change:** `user.home_area`, `user.country`, `user.language` and `conversation_msg` exist since migration 0001.
- **Limits:** messages of one user answered at the same time don't see each other in their history. A "Yes, delete" tap whose turn fails deletes nothing (the request was used up); the user asks again. `conversation_msg` keeps every exchange until "delete my data", with no expiry.

## Open questions for the owner

1. The setting values in `.env.example` (24 h, 20 messages, 3 tries, 60 s deadline, 180 s job timeout, 10 at once) are first choices.
2. `conversation_msg` keeps messages until "delete my data". Should it expire, like short-term memory?
3. Prompt `2026-11-21.1` changes the approved `2026-11-20.3` wording (Q3); approve for now, as with 0015?
