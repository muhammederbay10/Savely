# 0009: How the Instagram webhook stores shares and queues jobs

- Status: proposed with SAV-11. It takes effect when SAV-11's pull request is merged. The points under [For Muhammed](#for-muhammed) are choices made inside the ticket's scope that he may want to change.
- Date: 2026-11-17, revised 2026-11-18 after the first and second reviews (points 3–5, 7, 13 and 14)
- Superseded in part: point 7's time-only route and Q1 are replaced by record [0018](0018-model-decides-collection-hints.md) (SAV-39). The webhook binds a text to the share sent before it and queues it; the chat worker asks the decision model whether it names a collection. The share history and receipts stay.
- Ticket: SAV-11 (0008 is SAV-7's record on model choices)

## Context

`docs/ARCHITECTURE.md` §3.1 and §3.2 say what the webhook does: check the signature, skip duplicates, map the message to a `CaptureEvent` or a chat message, enqueue, and answer 200 at once. They don't say where the saved item is created, what a job looks like, or how "a text sent right after a share" is recognized. Three records already narrow the answers:

- **0002 and 0003:** what the payloads hold. A share and the text after it are two separate webhooks, 9 seconds apart in the recording, with nothing linking them but the sender and the time.
- **0004:** every saved item keeps the id of its message (`mid`).
- **0007:** an item is stored once per post id and once per message id. `post_url` and `platform_post_id` may be missing.

The tickets that run the jobs come later: the indexing pipeline (SAV-14) and the response agent (SAV-17). Their workers don't exist yet, so this ticket defines what they will find in the queue.

## Decision

1. **The webhook writes the saved item itself, and the job only indexes it.** `CaptureEvent.user_id` is our user, so the user has to be found before the event exists. The request therefore does three short pieces of work: find or create the user, insert the `saved_item` as `pending`, and queue one job. Each is a few milliseconds. Everything slow (downloads, models) runs in the job. The item's `saved_at` is the time the user sent the message, as Meta reports it.
2. **Duplicates are stopped by the unique keys of record 0007, not by a separate list of seen message ids.** The item is inserted with `ON CONFLICT DO NOTHING`. When nothing is inserted, the existing item is looked up by post id or message id.
3. **Every job has an id made from what it works on:** `index:<saved item id>` and `chat:<receipt id>` (see point 7). `arq` refuses a second job with an id it still knows, so two shares of one post are one indexing job.
4. **Database and Redis state first, queue second, and an error means "send it again".** Each message is handled in one database transaction that holds the user's row lock (point 13): the item is stored or the text is routed, and what the job needs is written to Redis. The job is queued after the commit, so a worker never sees an item before it exists. If Postgres or Redis fails, the endpoint answers 500 and Meta delivers the webhook again. Handling a webhook twice is safe:
   - the item already exists, so no second item is created;
   - its job is queued again only while the item is still `pending`. This covers a first delivery that stored the item and then failed to queue it;
   - a text follows the receipt of its first delivery (point 7), and its chat job is queued again only if the receipt says it wasn't queued.
5. **Job contract: ids in the job, data in a payload that "delete my data" can reach.** Workers register functions with these names and keyword arguments, all ids as text:

   | Job | Keyword arguments | arq queue | Payload, in the hash `savely:ingestion:<user_id>:payloads` |
   |---|---|---|---|
   | `index_saved_item` | `user_id`, `saved_item_id` | `arq:queue:indexing` | field `capture:<saved_item_id>`: the `CaptureEvent` as JSON |
   | `answer_chat_message` | `user_id`, `message_receipt_id` | `arq:queue:chat` | field `chat:<message_receipt_id>`: the `ChatMessage` as JSON |

   Each job name has its own queue (SAV-14, record 0012), so a worker reads only the jobs it can run. On one shared queue the indexing worker took chat jobs and failed them for good. The indexing worker's stuck-item sweeper may queue `index_saved_item` again, with the same job id, for an item no job is working on any more (record 0012).

   The indexing job needs the whole capture event because `saved_item` has no column for the media link: the signed image or video link exists only in the event, and the pipeline must download it before it expires. A share of the same pending item that was **sent** later replaces the payload, so the job gets the freshest link; an older message delivered again or late never replaces a newer payload. A worker that finds no payload stops, because the user was deleted. arq keeps a job's arguments in its queue and in the job's result; with ids only, neither holds a caption, a text or a link.
6. **A repeated share changes nothing on the item.** The first share's message id, caption and save time stay. Answers reply to the first message.
7. **The collection hint is a time rule, decided once per text.**
   - **Share history:** every share, also a repeated share of a post that is already saved, is recorded with the item it is stored as and the time the user sent it. The item's own message id and save time don't change (point 6). Webhooks can arrive out of order, so the history is kept by send time, not by arrival: a hint delivered after a later share still finds the share sent before it. Shares sent more than `INGESTION_STATE_TTL_SECONDS` before the newest one are dropped.
   - **Route:** a new text is the collection hint of the **last share sent before it**, when that share was sent at most `COLLECTION_HINT_WINDOW_SECONDS` before the text and its item has no collection yet. Otherwise it is a chat message; an older share in the window doesn't take over (one item per hint). The times are Meta's message times, not our clock.
   - **Receipt:** the route is kept as a receipt, keyed by the message id, **before** the item is changed or a job is queued. A repeated delivery follows the receipt and is never routed again, even if a delayed share arrived in between. A chat receipt also records whether its job was queued, so a delivery whose queuing failed is queued on the next delivery, once, and its route doesn't change.
   - The hint is stored in `saved_item.collection` exactly as the user wrote it ("For my movies collection"), and no chat job is queued for it.
8. **First contact takes a short lock.** Two first messages from one account can arrive together. A Postgres advisory lock on the account makes the second wait and find the first one's user, so an account never gets two users.
9. **One saved item per message.** If a message ever holds two shares, the first is captured and the rest are logged as a warning (`instagram_extra_shares_ignored`). No recorded message has two.
10. **Link or file is decided by the host.** A `url` on `instagram.com` is the post's public page and becomes `post_url`. Any other `url` is a file and becomes `media_url`. This holds for every attachment type, so a signed link can't end up in `post_url` even if Meta changes a payload.
11. **Answers Meta gets:** 200 when the webhook was handled, also when it held nothing for us (read receipts, reactions, the bot's own messages, the dashboard's test sample); 401 for a missing or wrong signature; 400 for a signed body that isn't a webhook object; 500 when Postgres or Redis failed.
12. **Nothing personal is logged.** Logs hold our own ids (user, saved item), the outcome and the attachment type. They never hold the sender's Instagram id, a message id, a caption, a text or a link. The API's database engine hides statement values in its errors for the same reason.
13. **Ingestion and "delete my data" take the same lock.** Ingestion locks the user's row (`FOR NO KEY UPDATE`) for the whole transaction of one message, and `delete_user_data()` locks it `FOR UPDATE`. A deletion therefore waits for the message being handled, and a message that arrives during a deletion waits and then finds no user: it is refused with a 500 (`UnknownUserError`), and Meta's next delivery starts a new user. Nothing is written to Redis for a user who is being deleted.
14. **Ingestion's state lives in Redis, per user, for `INGESTION_STATE_TTL_SECONDS`.** Three keys hold everything ingestion keeps about a user: the hashes `savely:ingestion:<user_id>:payloads` (point 5) and `:receipts` (field `<platform>:<message id>`), and the sorted set `:shares` (one entry per share, scored by its send time, point 7). Each write renews the key's lifetime. It must cover the time Meta keeps delivering a failed webhook again: up to 36 hours by the Graph API webhooks documentation (the Messenger Platform page says a subscription is switched off after 1 hour of failures). `.env.example` sets 36 hours. Redis is where the design keeps session state (§2, §5), so no table was added.
15. **"Delete my data" erases the Redis state too.** `delete_user_data()` now takes an eraser for the stores outside Postgres and calls it while it holds the user's lock, before it deletes any row. The ingestion eraser (`app/ingestion/erasure.py`) removes the user's indexing and chat jobs from their queues (and from arq's default queue, where jobs queued before the queues were split may be) with their retry state and kept results, then deletes the user's three hashes. If Redis fails, no row is deleted and the request can be repeated.

## For Muhammed

These follow the ticket and the earlier records, but each is a product choice you may see differently.

| # | Choice made here | Why, and what the other option is |
|---|---|---|
| Q1 | A question typed within the window after a share is taken as that share's collection, and gets no answer | The ticket asks for a time rule, and code can't tell "For my movies collection" from "Restaurants in Beşiktaş" without a language rule or a model. `.env.example` sets the window to 30 seconds (9 seconds was recorded). A shorter window loses slow typers; a longer one swallows more questions. SAV-14 could ask the decision model "is this a collection name?" instead |
| Q2 | The hint is stored as the user's whole sentence, not as "Movies" | Pulling the name out of the sentence needs a model or a rule for one language (record 0005 rules that out). The pipeline (SAV-14) can normalize it when it indexes the item |
| Q3 | Only the last share before the hint gets it | The ticket says "that item". If a user shares three reels and then types one hint, the first two get none |
| Q4 | Receipts and the share history live in Redis for `INGESTION_STATE_TTL_SECONDS` (36 hours in `.env.example`) | It covers Meta's documented redeliveries. A text delivered later than that is routed again. The other option is a Postgres table that keeps them forever, which changes ARCHITECTURE §4 (see Alternatives) |
| Q5 | `arq` is added to `pyproject.toml` | It is the queue named in the ticket and in SAV-22 §6, but it is a new line in the dependency list (it brings `redis` and `hiredis`) |
| Q6 | The legacy `share` fixture is synthetic | Meta never sent us a `share` attachment (record 0002, finding 7), so none could be recorded. The file follows Meta's documented shape and is named `synthetic_…` |

## Alternatives

- **Captions, texts and links inside the job arguments** (the first version). arq keeps the arguments in the queue and again in the job's result, and a worker running at the moment of a deletion writes that result afterwards. Ids in the job and the data in one per-user hash make erasure one `DEL` and leave nothing in a result.
- **A Postgres table for receipts and the share history.** It would make the routing decision and its receipt one transaction and keep receipts forever, but it adds a table that `docs/ARCHITECTURE.md` §4 doesn't have. The user's row lock gives the same safety with Redis, which the design names for session state. The cost is the lifetime: a webhook delivered again after `INGESTION_STATE_TTL_SECONDS` is routed again.
- **Hints by the item's save time** (the first version). A repeated share doesn't change the item, so a hint after it went to whichever other item was saved last.
- **Only the latest recorded share** (the second version). A hint delivered after a later share found only that later share and became a chat message. The history keeps every recent share by send time.
- **Queue the raw event and let a worker write the item.** The webhook would be even shorter, but a worker would have to exist in this ticket, `CaptureEvent.user_id` couldn't be filled by the adapter's caller, and a share would be invisible to search until a worker ran.
- **A list of seen message ids in Redis** (`SET NX` with a lifetime). It reads like "skip duplicate message ids", but it has to be written before the work and removed again when the work fails, or a failed webhook is lost on redelivery. The unique keys and job ids need no cleanup and can't disagree with the database.
- **Run the work after answering 200** (a background task). Faster by milliseconds, but a failure after the answer is never redelivered by Meta, so the share is lost.
- **Update the item on a repeated share** (new message id, new time), so that a hint after a repeated share finds it by the item's save time. The share history (point 7) does this without touching the item; updating it would move its save time and reply target, and an old webhook delivered late could move them back.
- **Give the hint to every share in the window.** It fits "three reels, one hint", but also labels an unrelated share that happens to be recent. Left as Q3.
- **Catch the unique-key error instead of locking on first contact.** It works, but the error path only runs when two requests overlap, which is hard to keep tested. The lock has one path.
- **Positional job arguments.** Keyword arguments keep a worker from reading them in the wrong order.

## Consequences

- **New settings:** `META_APP_SECRET`, `META_WEBHOOK_VERIFY_TOKEN` (both secrets), `COLLECTION_HINT_WINDOW_SECONDS` and `INGESTION_STATE_TTL_SECONDS` (at least the hint window), in `.env.example`, the README and `docker-compose.yml`. Like every setting they have no default, so the API and `alembic` refuse to start without them.
- **New dependency:** `arq` (Q5).
- **New code:** `app/adapters/`, `app/api/webhooks.py`, `app/ingestion/` and `app/db/session.py`. `app/ingestion/` is new in `docs/ARCHITECTURE.md` §6.
- **The API opens no connection at startup.** Postgres and Redis are first contacted by the first webhook, so `/health` still answers without them.
- **SAV-12 (Send API):** nothing is sent from the webhook. A confirmation or an "already saved" reply belongs to the jobs.
- **SAV-14 (pipeline):**
  - register `index_saved_item` with the arguments above, and move the item on from `pending`;
  - read the capture event from the payloads hash and download its `media_url` at once. No payload means the user was deleted: stop;
  - read `saved_item.collection` when classifying. The hint arrives seconds after the share, so it is on the row, not in the payload;
  - a collection-saved reel has no caption, post id or link, only the video.
- **SAV-17 (response agent):** register `answer_chat_message` and read the chat message from the payloads hash; no payload means the user was deleted. Storing the conversation (`conversation_msg`, Redis history) is not done here.
- **Duplicate jobs:** a text's chat job is queued at most once per receipt. Only one narrow case remains: the job was queued, and the step that marks the receipt failed. The next delivery then queues it again, and arq refuses the id only while it knows the job (queued, running, or its result kept: one hour by default). Keep the default for chat workers.
- **Until the workers exist,** queued jobs wait in Redis and expire after a day (arq's default), their payloads after `INGESTION_STATE_TTL_SECONDS`, and items stay `pending`. Sharing the post again queues it again.
- **A late delivery after the state's lifetime** finds no receipt and is routed again. With 36 hours this is past Meta's documented retries.
- **Tests with a real Redis** are skipped unless `TEST_REDIS_URL` is set, like the database tests (record 0007, P4).
- **Delete my data** erases the user's Redis state and queued jobs (point 15). What remains: a job queued in the moment after a message's commit and before a deletion holds only ids and finds no payload, and a job a worker is running during the deletion finishes and leaves a result with only ids. Stored videos are deleted by the ticket that stores them.
- **Not handled:** a message the user unsends (`is_deleted`) is ignored and its item stays. Message edits, reactions and read receipts are ignored.
