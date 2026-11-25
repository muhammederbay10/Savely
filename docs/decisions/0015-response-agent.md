# 0015: The response agent and its tools

- Status: proposed with SAV-17; it takes effect when SAV-17's pull request is merged. Muhammed decided all nine questions below in chat on 2026-11-20 ("Approve all nine"); evidence: SAV-17 comment `01a1fa0f-5999-77af-85b4-ef4de0e2afb7`, see [Owner decisions](#owner-decisions).
- Date: 2026-11-20
- Ticket: SAV-17

## Context

ARCHITECTURE §3.5 says the response agent (DeepSeek V4.1 Flash) understands a DM, calls tools and writes the reply, and never queries the database itself. Record 0004 says answers that list saved items are an intro plus one reply per result that quotes the saved reel, capped at about 5, with fallbacks. Record 0005 says replies work in any language and a correction moves one item without creating a synonym. Record 0011 says a language without built-in reply wording needs wording from the caller. Record 0014 gives search's result: `relaxed`, `level`, `fallbacks`, `incomplete`, `normalized` and `availability_gaps`.

Two parts of the ticket depend on tickets that have not started: `resolve_choice` needs SAV-15's "which one?" flow, and "the second one" needs the last shown results, which short-term memory (SAV-18) keeps. The owner decided who builds each (Q1, Q2, Q5).

## Decision

1. **The database is reached only through `app/agent/tools.py`.** `Toolbox.for_user(ToolContext(user_id, country))` binds the tools to the DM's sender; no tool takes a user id, and another user's item is answered exactly like a missing one. `tests/agent/test_agent_boundary.py` fails if the agent's loop, its prompts or the answer delivery import the database layer, search or the pipeline.
2. **Tools.** `search_items`, `near`, `get_item`, `update_item`, `set_status`, `request_data_deletion` and `confirm_data_deletion`. Arguments are Pydantic models with unknown keys refused, and search filters are checked against the content type by `plan_search`. A wrong argument, an unknown item or a missing country comes back to the model as an error it can act on (`ask_user_for: country`). The tools read the registry and never name a type or a filter: `near` searches the type that declares a `LOCATION` filter, `open_now` the filter whose source is `OPENING_HOURS`. `set_status(item_id, status, entity_position?)` writes `user_entity` for the item's place or title; an item without an entity, such as a study post, keeps its status on `saved_item.user_status` (migration 0003), separate from the indexing `status` and deleted with the item (Q6). `get_item` shows both. `resolve_choice` is SAV-15's (Q1).
3. **The final answer is JSON** (`language`, `reply`, `notice`, `item_ids`) and is checked before it is accepted: only items a tool returned this turn or in the working set, a notice when a search fell back, and a reply that fits in one DM. A failed check is sent back to the model as a system message. The loop is capped at `AGENT_MAX_STEPS`, and the last call has `tool_choice="none"`, so it ends with an answer or `AgentStepLimitError`.
4. **Notices** come from search's result, not from the model's judgement: `backup_mode` when a step went to L1, `broader_results` when a result was found only at L2, `may_be_incomplete` when the time budget ran out. The model writes the sentence in the user's language; the agent makes sure there is one.
5. **Language.** The model replies in the language of the user's message and names it with a BCP 47 tag. For a language without built-in wording (record 0011), the agent asks the model once for the `ReplyWording` fields in that language, translated from the English ones. Wording that breaks the placeholder rules raises `AgentWordingError`; the user is never answered in another language.
6. **Delivery** (`app/agent/replies/answers.py`): the intro, then one message per result, at most `AGENT_MAX_RESULTS_PER_ANSWER`. Per result: a reply quoting the saved message with the Maps or TMDB link; if Meta refuses it (`MessengerRequestError`), the post link as a preview card; then the name with the link. A closed window stops the answer. Any other messenger error is raised and not retried as another step, so a message that may have been delivered is not sent twice. **Limit (Q7):** Meta answers a reply with 200 and a message id whether or not Instagram shows the quoted reel, so only a refused reply falls back; a reply that was accepted but shows no quote (an old or deleted message) can't be detected.
7. **Working set (Q2).** `AgentTurn.working_set` is the ordered ids of the user's last answer, and `AgentAnswer.working_set` the ordered ids of this one, shown and "show more" alike. The agent shows the model the list by number, so "the second one" is resolved by the model to an id. The agent keeps nothing between messages; SAV-18 stores the list in Redis and passes it in.
8. **Corrections** (`app/pipeline/vocabulary/corrections.py`) normalize the user's wording through the vocabulary's normal steps, put it on that item only with `user_set: true` in `raw_values`, and replace the path's provenance in the normalize step. The alias that produced the old value gets `correction_count + 1` and is added to the corrected path's provenance as `corrected_alias_ids` (the run's JSON output, no schema change). `verify_alias` refuses a candidate with any correction. A verified alias is flagged for review (log line `vocabulary_alias_flagged_for_review`) once **different users** corrected it at least `VOCAB_ALIAS_REVIEW_MIN_USERS` times, counted as distinct `saved_item.user_id` over that provenance, like the verification query; one user's repeated corrections count once (Q4). The evidence belongs to the items, so "delete my data" removes it.
9. **Links.** The Maps or TMDB link is read from `entity.data["link"]`. The resolvers (SAV-15) store it there: Google Places returns `googleMapsUri`, and the TMDB page URL is built from the title's id. The agent builds no URL itself.
10. **"Delete my data" from the DM** (owner update of 20 Nov 2026; revised after interim review round 1 and Q9). The model can never delete on its own word; code decides. `request_data_deletion` stores nothing: it makes the answer the fixed `deletion_confirmation` question with one **"Yes, delete" quick reply** whose payload is the fixed `DELETION_CONFIRM_PAYLOAD`, and prepares the fixed `deletion_done` message in the same language (`AgentAnswer.deletion_confirmation`). The request is planned right before the question is sent and stored in Redis (`app/agent/deletion_requests.py`) only after the whole answer was sent (`app/agent/send.py`), with its deadline, language and "done" text. The deadline is `DATA_DELETION_CONFIRM_SECONDS` after **sending the question begins**, so the send time shortens the window and storing the request late never lengthens it (owner decision of 22 Nov 2026, SAV-18 comment `01a20456-384c-7d33-a2c5-3ba8c67647b0`, record 0016; it replaces "after delivery"); a question that failed to generate, translate or send leaves nothing to confirm. `UserTools.begin_turn` takes the request out at the start of the user's **next** turn, so "no" or any other message uses it up and a same-turn request never counts (SAV-18 binds "next" to ingestion's message numbers, record 0016). The Instagram adapter keeps `message.quick_reply.payload` as `ChatMessage.quick_reply_payload`, and `confirms_deletion(payload)` is true only for the button's payload; the chat worker (SAV-18, Q5) sets `AgentTurn.deletion_confirmed` from it. `confirm_data_deletion` takes no arguments. Without that confirmation, a typed "yes" (or a wrong call) while the question is open asks it again with its button and deletes nothing; with it, the tool erases only if the deadline has not passed, checked right before erasing and again by the eraser once it holds the user's row lock, which it may have to wait for. It reads the sender's platform ids, then runs `UserDataEraser` (`app/db/deletion.py`): every store outside Postgres in order (ingestion's Redis state and jobs, the pending request, the media store), then `delete_user_data`, in one transaction under the user's row lock. A successful erasure ends the turn at once with the prepared "done" message and `deleted_recipients`: no model call, translation or other tool runs after it. Until SAV-18's worker sets the flag, DM deletion fails closed.
11. **Prompt wording** lives in `app/agent/prompts.py` as `AGENT_PROMPTS`, version `2026-11-20.3`, approved by the owner for now (Q3). The tests use a fake model, so they check the loop and the tools, not the model's quality; scoring and before/after numbers come with the evaluation set (SAV-19/20).

## Owner decisions

Muhammed decided all nine questions in chat on 2026-11-20 ("Approve all nine"). Evidence: SAV-17 comment `01a1fa0f-5999-77af-85b4-ef4de0e2afb7`, passed on by Jarvis in `01a1fa11-2c91-787f-9754-9fbe0fca0147`. None is open.

| # | Question | Decision | Where it is done |
|---|---|---|---|
| Q1 | Who builds `resolve_choice` | SAV-15 builds it: it stores each pending "which one?" question with its item id and candidate ids; the quick-reply payload is `choice:<item id>:<candidate number>`; `resolve_choice(item_id, option)` links the chosen candidate or marks "none of these" | SAV-15 (in its description). Not in SAV-17 |
| Q2 | Where the working set lives | An input and output of the agent (point 7); SAV-18 stores it in Redis | SAV-17: the contract. SAV-18: storage |
| Q3 | Prompt `2026-11-20.3` | Approved for now; scored on the evaluation set, with before/after numbers, when SAV-19/20 exist | SAV-19/20 |
| Q4 | Review flag for verified aliases | `corrected_alias_ids` in the corrected path's normalize-step provenance, no schema change; flag on distinct `saved_item.user_id`. No count-only shortcut | SAV-17 (point 8, `VOCAB_ALIAS_REVIEW_MIN_USERS`) |
| Q5 | Who builds the chat worker | SAV-18, with the history and working set; it also routes the deletion button's payload and adds its stores to `UserDataEraser` | SAV-18 (in its description) |
| Q6 | `set_status` shape; status of items without an entity | `set_status(item_id, status, entity_position?)`; an item without an entity keeps its status on itself, deleted with it | SAV-17 (point 2, `saved_item.user_status`, migration 0003) |
| Q7 | Detecting a missing quote | Accepted: fall back only when Instagram refuses the reply; the limit is noted | SAV-17 (point 6) |
| Q8 | Where the Maps/TMDB link is stored | `entity.data["link"]`, written by SAV-15's resolvers | SAV-15 (in its description); SAV-17 reads it |
| Q9 | How a deletion is confirmed | A "Yes, delete" quick reply with a fixed payload; the adapter reads `message.quick_reply.payload`; only that payload, on the sender's next message after the delivered question, sets `deletion_confirmed`. A typed "yes" shows the question again | SAV-17: template, button, adapter, checks (point 10). SAV-18: the worker routing that sets the flag |

## Alternatives considered

- **A "reply" tool for the final answer** instead of a JSON output schema: one more tool for the model to learn, and the answer would still need the same checks.
- **Trusting the model to write notices:** a model that forgets one would hide that search fell back, which §3.6 forbids.
- **Showing the hits of the last search automatically** instead of the model's `item_ids`: the model could not drop a wrong hit or order results from two searches.
- **Treating every Send API error as a reason to fall back:** a timeout or 5xx may have delivered the message, so a fallback would duplicate it (record 0011).
- **A fixed English result line for languages without wording:** breaks record 0005 §9.

## Consequences

- **SAV-15:** builds `resolve_choice` (Q1), stores `entity.data["link"]` (Q8), and fetches the rows in `AgentAnswer.availability_gaps`.
- **SAV-18:** builds the chat worker (Q5): stores `AgentAnswer.working_set` and the conversation and passes them in `AgentTurn`, stores the user's country once asked, sets `AgentTurn.deletion_confirmed` only when `confirms_deletion(message.quick_reply_payload)` holds, calls `send_answer`, and adds its Redis keys to `UserDataEraser`'s stores.
- **SAV-19/20:** score prompt `2026-11-20.3` and report before/after numbers (Q3).
- **Schema:** `saved_item.user_status` (migration 0003, Q6). `ChatMessage.quick_reply_payload` is a new optional field of the adapter's output.
- **New settings:** `AGENT_MAX_STEPS`, `AGENT_MAX_RESULTS_PER_ANSWER`, `AGENT_MAX_OUTPUT_TOKENS`, `VOCAB_ALIAS_REVIEW_MIN_USERS`, `DATA_DELETION_CONFIRM_SECONDS`.
- **Costs:** one text-model call per step; one more per answer only for a language without built-in wording.

## Update: tool steps and the answer step (SAV-38, 24 Nov 2026)

**Context.** In the owner's live test on `e30b152`, the bot answered "What did i save?" with "Let me pull up your saves." and no items; the log had `agent_answered steps=3 results=0` and no `agent_tool_called`. Replayed through the real `ResponseAgent`, `deepseek/deepseek-v4.1-flash` on CoreWeave never called a tool in 3 of 3 tries when a request carried both the tools and the answer's `response_format`; it went straight to the final JSON, once with "Here's what you saved most recently." and no items. With the same messages and tools but no `response_format`, it called `search_items` in 3 of 3 tries.

**Decision.**
- **Tool steps send the tools and no answer format** (`tool_choice="auto"`). When the model stops calling tools, a reply that is already a valid answer object is taken as it is; otherwise the next call is the **answer step**: the answer format, with tool calls forbidden (`tool_choice="none"`; the tools are still listed because the conversation holds tool calls). The last step is always an answer step, as before.
- **The answer carries `outcome`**: `results` (the reply shows the items in `item_ids`, which must not be empty), `no_results` (a search found nothing; `item_ids` is empty) or `other` (a question, a greeting, a confirmation). The model reports it, so the check works in any language and never reads the reply's words. An answer whose `outcome` does not match `item_ids` is sent back like any other failed check. The deletion question replaces the reply, so it is not checked there.
- **Where a failed answer goes:** a malformed answer is asked for again in the answer step; any other failed check goes back to a tool step, so the model can search before it answers again.
- The prompt (`2026-11-24.1`) tells the model to call the tools before it answers, never to promise to look, and how to set `outcome`.
- **Real-model check:** `tests/live/test_response_agent_live.py`, opt-in, run by the owner before a push (SAV-22, 24 Nov 2026 update).

**Alternatives considered.**
- **`tool_choice="required"` with a "reply" tool for the answer:** one call fewer, but the live evidence covers only "tools without a format", and whether the pinned provider honours `required` with this model was not checked. It also changes the answer's contract (see the alternatives above).
- **Matching words like "here's what you saved" in the reply:** works in one language only; ruled out by decision record 0005.

**Consequences.** An answer the model writes in prose costs one more call (the answer step), within `AGENT_MAX_STEPS`. The answer schema has a new required field, `outcome`.

### Long answers (SAV-38 live check, 25 Nov 2026)

**Context.** In the owner's combined live run (SAV-39 head `cce6839`, which contains `2420a7d`), all 12 runs called a search tool and 11 answered. In one "What did i save?" run the answer step listed all 11 item ids; its JSON was cut off at the output limit (400 tokens in the test, 600 in `.env.example`) and the turn ended in `SchemaOutputError`. An answer's size grew with every id the model wrote, so a user with many saves could hit any fixed limit. The owner decided to fix it, not to raise the limit.

**Decision.**
- **The answer lists at most `AGENT_MAX_RESULTS_PER_ANSWER` items**: the answer format's `item_ids` has `maxItems` and says so, the prompt says so, and the code shows only that many whatever the model lists. The answer stays a few hundred tokens at most.
- **The rest are kept for "show more"**, in order: the ids the answer listed beyond the cap, then the unlisted results of the searches the answer presents. Each shown item belongs to the latest search that returned it, so a broad search replaced by a narrower one (all saves, then restaurants) adds nothing, while an answer drawn from two searches keeps both (SAV-38 review, round 4). When the answer continues the last results without a new search, the last results not shown yet are kept instead. An answer that shows nothing keeps nothing, so the previous working set stays.
- **A cut-off answer is asked for again.** The answer step's reply that is no JSON (`SchemaOutputError`, which is how a cut-off at the output limit arrives) goes back to the model with "answer again, shorter, with at most N ids", in the answer step. The first such retry in a turn gets one call beyond `AGENT_MAX_STEPS`, so a cut-off at the last step still recovers; a second one uses the normal steps, and the step limit still ends the turn. A prose reply cut off in a tool step (`finish_reason` `length`) is dropped instead of being kept as the answer's draft. Ownership, notices, the answer checks and the deletion flow are unchanged.
- Prompt `2026-11-25.1`. The live check runs with the agent settings of `.env.example` and checks the cap and "show more" for "What did i save?".

**Alternatives considered.**
- **A larger `AGENT_MAX_OUTPUT_TOKENS`:** moves the limit without removing it; ruled out by the owner.
- **Repairing a cut-off JSON** (closing the brackets of a partial answer): the reply itself may be cut, and a repaired answer could not be checked like a whole one.
- **Asking again only within `AGENT_MAX_STEPS`:** a cut-off at the last step, the forced answer step, would still end in the error message.
- **Leaving the cap to the model only:** the model ignored the old prompt's cap; the code must not trust it.

**Consequences.** "Show more" can list search results the answer chose not to show. `maxItems` in the answer format is checked only by the opt-in live run; if the pinned provider ignores it, the code's cap and the retry still hold. A turn makes at most `AGENT_MAX_STEPS` + 1 text-model calls (plus the wording translation).
