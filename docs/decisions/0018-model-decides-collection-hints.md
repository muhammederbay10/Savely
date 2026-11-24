# 0018: The decision model decides whether a text after a share is a collection name

- Status: accepted. Muhammed dropped the time-only rule by creating SAV-39 (24 Nov 2026) and approved the three proposal points (worker-side routing, the window as an upper bound with unsure or failed decisions going to chat, and the confirmations) on SAV-39 the same day.
- Date: 2026-11-24
- Ticket: SAV-39. Supersedes the time-only hint rule of record 0002 (consequence on hints) and record 0009 (point 7, "Route", and Q1). Everything else in 0009 point 7 stays: the share history by send time, one receipt per text, one item per hint.

## Context

In Muhammed's live test on 24 Nov 2026 he shared a reel and asked "What did i save?" 20 seconds later. The bot never answered: the text was within `COLLECTION_HINT_WINDOW_SECONDS` (30 s) of the share, so it became the reel's `collection_hint`. Record 0009 had flagged this (Q1): code can't tell "For my movies collection" from "Restaurants in Beşiktaş" without a language rule (ruled out by record 0005) or a model. Real users share and ask right away.

The webhook must answer Meta fast and calls no model (ARCHITECTURE §3.2), and a repeated delivery must be routed the same way and labelled on the right share (record 0009).

## Decision

1. **The webhook only binds.** For a new text it looks for the user's last share sent before it, within `COLLECTION_HINT_WINDOW_SECONDS`, whose item has no collection yet (the same send-time lookup as 0009). A tap on one of the bot's buttons (a quick reply) answers the bot and is never bound. With one, the text's receipt gets the route `undecided` and that item's id; without one, the route `chat`. Every text gets its payload and a chat job (`answer_chat_message`). The receipt is kept before the job is queued, so a repeated delivery is bound the same way and queued once.
2. **The chat worker decides, once.** Before the response agent runs (`app/ingestion/text_routing.py`):
   - an `undecided` receipt whose item has a collection by now is `chat`, with no model call;
   - otherwise the decision model (Gemini Decision 1 through the `Classifier` interface) answers one yes/no question: does the text only name a collection, folder or list for the post just shared? (`ROUTING_PROMPTS`, versioned). At or above `COLLECTION_HINT_MIN_PROBABILITY` it is a `collection_hint`;
   - below the threshold, an unanswered question or a failed call (`ModelError`) is `chat`. A missed label costs little; a swallowed question looks broken;
   - under the user's row lock, the decision is written to the receipt and, for a hint, the item gets the text as its collection (`set_collection_if_missing`). A later try, or an overlapping one, reads the receipt and follows it; the model is never asked twice for one text. The model is never called while the lock is held.
   - The receipt is written before the database commits. If the commit fails, the next try follows the receipt and sets the collection again, which changes nothing once it is set.
3. **The window is only an upper bound.** A text sent later than the window after the share is `chat` at the webhook, and the worker makes no model call for it.
4. **The user sees what happened, in one message when possible.**
   - The indexer reads the item's collection and marks its confirm step `started` (`enrichment_run`) in one transaction under the owner's row lock, then sends. A collection set before that is part of the save confirmation: "Saved as café · in Food" ("… · Food koleksiyonunda").
   - The chat worker reads that mark under the same lock when it sets the collection. If the confirmation had started, it keeps `confirm_collection` in the receipt and sends one short message of its own: "Saved to Food" ("Food koleksiyonuna kaydedildi"), in the user's profile language, English while it is unknown, like the save confirmation (SAV-14). Otherwise it sends nothing.
   - The collection confirmation is sent once: the reply mark is written right before sending (`SessionMemory.start_reply`), so a later try sends nothing, and a failed send is not tried again.
   - A hint is not part of the conversation the response agent reads, as before.
5. **"Delete my data"** takes the same user row lock. A user deleted before the job runs has no payload or receipt left: the job stops with no model call. Every text's receipt names its chat job, so the eraser removes every queued job of the user, whatever its route.
6. **New setting:** `COLLECTION_HINT_MIN_PROBABILITY` (chat worker, `ChatSettings`), from 0 (excluded) to 1. `.env.example` sets 0.7, to be tuned on the live routing check.

## Alternatives considered

- **Decide in the webhook.** Meta needs a fast 200, and a model call there would make a slow or failed model delay or fail the webhook. Rejected in the SAV-39 proposal.
- **Keep the time rule, shorten the window.** A shorter window loses slow typers and still swallows quick questions.
- **A word list per language** ("for my", "koleksiyon"…). Ruled out by record 0005.
- **Hold every save confirmation until a possible label arrives.** It would delay every confirmation by the window for the few shares that get a label. The proposal chose a separate short confirmation for the rare case where the save was already confirmed.
- **Ask the text model (DeepSeek) to extract the name.** The question is a fixed choice with a probability; the decision model gives that directly and cheaply. The hint is still stored as the user's whole text, as in 0009 (Q2).

## Consequences

- A question right after a share is answered. Every text after a share within the window costs one decision call and its latency before the answer.
- A collection name is confirmed: in the save confirmation when it was not sent yet, otherwise in one extra short message.
- The receipt has a new route (`undecided`) and a new field (`confirm_collection`, default `false`); receipts written before this change still load. `IngestionOutcome.COLLECTION_HINT` is gone: the webhook only reports `chat_queued` or `chat_already_queued`, and logs the route. The chat worker logs `text_route_decided` with the route, the reason, the probability and the prompts version, never the text.
- The confirm step of `enrichment_run` has a new status, `started`, written before the confirmation is sent.
- Model behaviour changed, so SAV-22's opt-in live check applies: `tests/live/test_collection_routing_live.py` asks the real decision model about 24 texts in four languages, half collection names. Muhammed runs it locally with SAV-38's check (`docs/development.md`, "Live routing check").
- `docs/ARCHITECTURE.md` §3.1–3.2, `docs/development.md` and `.env.example` describe the new route.
