# Development

How to set up, run and test Savely, and how its parts work. The product overview is in the [README](../README.md).

- Design: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Rules for contributors and agents: [`AGENTS.md`](../AGENTS.md)
- Decision records: [`docs/decisions/`](decisions/)

This repository currently holds the project skeleton (a FastAPI app with a health endpoint, the database schema with its migrations, a local Docker stack and the development tooling), the content-type registry, the Instagram webhook, which stores what users share with the bot and queues the work that follows, and the indexing pipeline, which works out what each saved post is about.

## Requirements

- Python 3.12
- Docker with Compose v2 (`docker compose version`)
- `make`
- ffmpeg, which includes ffprobe, to run the indexing worker on your machine (`brew install ffmpeg` on a Mac). The Docker image installs its own.

## Setup

```bash
git clone https://github.com/muhammederbay10/Savely.git
cd Savely

cp .env.example .env            # local configuration, git-ignored

python3.12 -m venv .venv
source .venv/bin/activate
make install                    # the app plus ruff, mypy and pytest
```

The values in `.env.example` work as they are for local development. To receive real webhooks from Meta, set the two `META_*` values as described under [Instagram webhook](#instagram-webhook).

## Run

```bash
make up
```

This builds and starts five services and waits until all of them are running (and healthy, for the three that have a health check):

| Service | What it is | Address on your machine |
|---|---|---|
| `api` | The FastAPI app | http://localhost:8000 |
| `worker` | The indexing worker (see [Indexing pipeline](#indexing-pipeline)) | none |
| `chat_worker` | The chat worker: answers DMs (see [Chat worker](#chat-worker)) | none |
| `db` | Postgres 16 with PostGIS and pgvector | `localhost:5432` |
| `redis` | Redis 7 | `localhost:6379` |

Check that the API is up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Create the database tables (once after the first `make up`, and again after pulling new migrations):

```bash
make migrate
```

Useful commands:

```bash
docker compose logs --follow api   # API logs, one JSON object per line
docker compose logs --follow worker # indexing logs
docker compose logs --follow chat_worker # chat logs
make down                          # stop everything, keep the database data
docker compose down --volumes      # stop everything and delete the local database
```

If a port is already in use, change `API_PORT`, `POSTGRES_PORT` or `REDIS_PORT` in `.env`.

### Local data from before the rename

The product was called SavedAgent until 18 November 2026. Local data from that time is development data only, and nothing migrates it:

- `.env.example` now names the Postgres user and database `savely`. A `.env` copied earlier still says `savedagent` and keeps working with its existing database; the commands in this guide then need its old user. If you switch `.env` to the new values, the existing database volume still has only the old user and database, so Postgres refuses the new ones.
- Ingestion's Redis keys are now named `savely:ingestion:…`. Keys under the old `savedagent:ingestion:…` names are no longer read and expire after `INGESTION_STATE_TTL_SECONDS`.

To start clean, delete both with `docker compose down --volumes` (this deletes all local database and Redis data), then run `make up` and `make migrate`.

### Run the API outside Docker

For auto-reload while you edit code, run only the database and Redis in Docker and the API on your machine:

```bash
docker compose up --detach --wait db redis
make run
```

`make run` reads `.env` and serves on http://localhost:8000.

## Test

```bash
make check
```

`make check` runs the four checks that must pass before any push. You can also run them one at a time:

| Command | Runs |
|---|---|
| `make fmt` | `ruff format .` (rewrites files) |
| `make lint` | `ruff check .` |
| `make typecheck` | `mypy app` |
| `make test` | `pytest` |

Tests never call external services.

Two tests run the real `ffmpeg` and `ffprobe` on a clip they generate. They are **skipped** when the programs are not installed, and pytest lists them with the reason. Every other pipeline test uses a stand-in for them.

### Database tests

The tests for the schema and the migrations need a real Postgres with PostGIS and pgvector. They are **skipped** unless `TEST_DATABASE_ADMIN_URL` is set, and pytest lists every skipped test with the reason. To run them, start the `db` service and point the variable at its `postgres` maintenance database:

```bash
docker compose up --detach --wait db
TEST_DATABASE_ADMIN_URL=postgresql+asyncpg://savely:change-me@localhost:5432/postgres make test
```

Each run creates its own databases named `savely_test_<random>` on that server and drops them at the end. It never reads or changes the `savely` database you develop against. The user in the URL must be allowed to create databases and extensions, which the Compose user is.

### Redis tests

The ingestion tests keep their state in a real Redis, and some check what arq stores there. They are **skipped** unless `TEST_REDIS_URL` is set; most of them also need `TEST_DATABASE_ADMIN_URL`. To run them, start the `redis` service and point the variable at it:

```bash
docker compose up --detach --wait redis
TEST_REDIS_URL=redis://localhost:6379/0 make test
```

The tests use new random ids for every user and job, so they never touch each other's keys, and keys they leave expire. Use a Redis that holds nothing else: one test reads every key to check that "delete my data" erased a user's texts. No worker should be reading that Redis while the tests run. Both variables can be set in the same run.

## Instagram webhook

Users share posts and reels with the bot account by direct message, or save them into a collaborative collection the bot belongs to. Meta sends each message to the API as a `messages` webhook.

| Endpoint | What it does |
|---|---|
| `GET /webhooks/instagram` | Meta's verification handshake. Returns `hub.challenge` when `hub.verify_token` equals `META_WEBHOOK_VERIFY_TOKEN`, and 403 otherwise |
| `POST /webhooks/instagram` | Receives a webhook. Checks `X-Hub-Signature-256` against the raw body with `META_APP_SECRET` (401 if it is missing or wrong), stores the share or routes the text, queues a job and answers 200 |

What happens to a message ([`app/adapters/instagram.py`](../app/adapters/instagram.py), decision records 0002, 0003 and 0009):

| Message | Result |
|---|---|
| A reel shared by DM (`ig_reel`) | A saved item with the reel's id, public link and caption |
| A post shared by DM or saved into the collection (`ig_post`, or the older `share`) | A saved item with the post's id (`ig_post` only) and caption. The signed image link is passed to the indexing job and not stored |
| A reel saved into the collection (`unsupported_type`) | A saved item with only its message id. The signed video link is passed to the indexing job and not stored |
| A text sent at most `COLLECTION_HINT_WINDOW_SECONDS` after the user's last share before it (also a repeated share of a saved post, and also when a later share arrived first) | Stored as that share's item's collection, e.g. "For my movies collection", if it has none yet. No chat job |
| Any other text | A chat job for the response agent |
| Anything else (read receipts, reactions, the bot's own messages, other attachments) | Ignored, and logged without its content |

- **First contact:** the first message from an Instagram account creates its `user` and `platform_account` rows.
- **Message id:** every saved item keeps the id of the message it came in with (`source_message_id`), so answers can reply to it (decision record 0004).
- **One item per post and per message:** sharing the same post twice, or a webhook that Meta delivers twice, creates one saved item. Two reels saved into the collection have no post id, so they stay two items (decision record 0007).
- **Jobs:** new items are queued as `index_saved_item` and texts as `answer_chat_message` on Redis, through `arq` ([`app/ingestion/jobs.py`](../app/ingestion/jobs.py)). A job holds only ids; the caption, text and media link it works on are in a per-user payload hash ([`app/ingestion/session_state.py`](../app/ingestion/session_state.py)). Each job name has its own arq queue (`arq:queue:indexing`, `arq:queue:chat`), so a worker only takes the jobs it can run. The indexing worker runs `index_saved_item` (see [Indexing pipeline](#indexing-pipeline)). The chat worker runs `answer_chat_message` (see [Chat worker](#chat-worker)).
- **Routing is decided once:** each text's route is kept as a receipt by its message id before anything changes, so a repeated delivery is routed the same way, also when a delayed share arrived in between.
- **Ingestion state:** payloads, text receipts and each user's recent shares stay in Redis for `INGESTION_STATE_TTL_SECONDS`, and "delete my data" erases them.
- **Errors:** if Postgres or Redis is down, the endpoint answers 500 and Meta delivers the webhook again. Handling a webhook twice is safe.

### Connect the Meta app

Meta needs a public HTTPS address, so local development uses a tunnel.

1. In `.env`, set `META_APP_SECRET` to the app's secret (Meta app dashboard → **App settings → Basic**) and `META_WEBHOOK_VERIFY_TOKEN` to any long random text.
2. Start the stack and a tunnel:
   ```bash
   make up && make migrate
   ngrok http 8000
   ```
3. In the Meta app go to **Use cases → Customize → API setup with Unified login → Configure webhooks**. Set the callback URL to `https://<your-tunnel>/webhooks/instagram` and the verify token to your `META_WEBHOOK_VERIFY_TOKEN`, then subscribe to `messages`.
4. In **Generate access tokens**, generate a token for the bot account, then switch **Webhook Subscription** off and on again. No direct message arrived before this step (decision record 0002, finding 10).
5. Send the bot a reel from an Instagram Tester account. `docker compose logs --follow api` shows `capture_ingested`.

The dashboard's **Test** button sends a sample in another shape than real messages. It is answered with 200 and ignored.

## Database

The schema is defined twice, and a test keeps the two in step:

- [`app/db/models.py`](../app/db/models.py): SQLAlchemy models, used for queries.
- [`app/db/migrations/versions/`](../app/db/migrations/versions/): Alembic migrations, the only thing that changes the database.

"Delete my data" is `delete_user_data()` in [`app/db/deletion.py`](../app/db/deletion.py). It deletes one user's rows and lowers the item counts of the shared vocabulary. It also takes an eraser for the stores outside Postgres: pass `IngestionDataEraser` from [`app/ingestion/erasure.py`](../app/ingestion/erasure.py), which removes the user's queued jobs and ingestion's Redis state. Ingestion and deletion lock the same user row, so they never overlap. The shared tables (entities, streaming availability, vocabulary and aliases) keep their rows.

| Command | What it does |
|---|---|
| `make migrate` | Apply every migration (`alembic upgrade head`) |
| `alembic downgrade base` | Drop every table the migrations created. **This deletes the data in them** |
| `alembic revision -m "add x"` | Start a new, empty migration to fill in by hand |
| `alembic check` | Fail if the models and the database differ |

Alembic reads `DATABASE_URL` and `EMBEDDING_DIMENSION` through `app/config/base.py`, from `.env` or the environment. Run it from the repository root.

The first migration creates the two embedding columns (`saved_item.embedding` and `vocabulary.embedding`) with `EMBEDDING_DIMENSION` components. The embedding model is `openai/text-embedding-3-large` at **1024 dimensions** (Muhammed's choice in SAV-7), so set `EMBEDDING_DIMENSION=1024`, as `.env.example` does. Both columns are then `vector(1024)`, and the database rejects a vector of any other length.

The dimension stays a setting, so another model can be tried on a fresh database. To change it on a database that is already migrated, write a new migration that resizes both columns, and embed every saved item and vocabulary value again.

## Spikes

[`scripts/spikes/`](../scripts/spikes/README.md) holds small experiments that back a decision record. `model_choices` compares speech-to-text and embedding models through OpenRouter (SAV-7, [record 0008](decisions/0008-model-choices.md)). By default it sends nothing; a live run costs money, reads `OPENROUTER_API_KEY` from the environment and needs the owner's approval first.

## Configuration

All configuration comes from environment variables. Locally they live in `.env`, and `.env.example` lists every variable. The variables of the model clients have their own table under [Model settings](#model-settings), those of search under [Search](#search), and those of the response agent under [Response agent](#response-agent).

| Variable | Used by | Meaning |
|---|---|---|
| `LOG_LEVEL` | API | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `DATABASE_URL` | API | Postgres URL, e.g. `postgresql+asyncpg://user:password@host:5432/db` |
| `REDIS_URL` | API | Redis URL, e.g. `redis://host:6379/0` |
| `EMBEDDING_DIMENSION` | API, migrations | Length of the embedding vectors, 1 to 2000. Must match the embedding model: `1024` for `openai/text-embedding-3-large` (see [Database](#database)) |
| `META_APP_SECRET` | API | The Meta app's secret. Every webhook is signed with it. **Secret** |
| `META_WEBHOOK_VERIFY_TOKEN` | API | The text Meta repeats in the verification handshake. You choose it and enter it in the Meta app. **Secret** |
| `COLLECTION_HINT_WINDOW_SECONDS` | API | A text sent this many seconds or less after a share becomes that item's collection hint. A whole number above 0 |
| `INGESTION_STATE_TTL_SECONDS` | API | How long ingestion keeps a user's job payloads, text receipts and recent shares in Redis. At least the hint window; 129600 (36 hours, Meta's documented retry period) in `.env.example` |
| `INSTAGRAM_GRAPH_API_URL` | DM replies | Graph API base URL with its version, e.g. `https://graph.facebook.com/v27.0` (see [DM replies](#dm-replies)) |
| `INSTAGRAM_ACCOUNT_ID` | DM replies | The bot's Instagram professional account id |
| `INSTAGRAM_ACCESS_TOKEN` | DM replies | The bot's access token for the Send API. A secret |
| `INSTAGRAM_REPLY_WINDOW_HOURS` | DM replies | How long after the user's last message the bot may reply: more than 0 and at most `24` (Instagram's limit) |
| `SEND_API_TIMEOUT_SECONDS` | DM replies | Time limit of one Send API request |
| `SEND_API_MAX_ATTEMPTS`, `SEND_API_RETRY_BASE_DELAY_SECONDS`, `SEND_API_RETRY_MAX_DELAY_SECONDS` | DM replies | Retries when Meta throttles a message or the connection never opens: total attempts, first wait, longest wait |
| `TEST_DATABASE_ADMIN_URL` | Tests only | Postgres server for the database tests (see [Database tests](#database-tests)). Not read by the API |
| `TEST_REDIS_URL` | Tests only | Redis server for the Redis tests (see [Redis tests](#redis-tests)). Not read by the API |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Docker Compose | Credentials and database name for the `db` container |
| `API_PORT`, `POSTGRES_PORT`, `REDIS_PORT` | Docker Compose | Ports published on `127.0.0.1` |

The API settings are defined in [`app/config/`](../app/config/): `Settings` for the API (`base.py`) and `MessengerSettings` for DM replies (`messenger.py`), which is only read when the bot sends a DM, so `/health` and the migrations don't need the token. None of them has a default, so the API refuses to start when one is missing or invalid. The error names the variable but never prints its value, because a URL can contain a password. The two secrets are also hidden when the settings are printed. Under Docker Compose, `DATABASE_URL` and `REDIS_URL` are built in `docker-compose.yml` from the `POSTGRES_*` values, so the two in `.env` are only used by `make run`.

## DM replies

[`app/agent/replies/messenger.py`](../app/agent/replies/messenger.py) sends the bot's DMs through the Instagram Send API, and [`app/agent/replies/templates.py`](../app/agent/replies/templates.py) holds the fixed wording (decision records [0004](decisions/0004-reply-to-saved-reels.md) and [0011](decisions/0011-dm-replies.md)). The indexing pipeline sends the save confirmation with them; the response agent (SAV-17) will send the rest.

```python
import httpx

from app.agent.replies.messenger import InstagramMessenger
from app.agent.replies.templates import ResultLine, reply_wording, result_reply
from app.config.messenger import get_messenger_settings

wording = reply_wording(user.language, localized=agent_wording)  # see "Languages" below
async with httpx.AsyncClient() as http_client:
    messenger = InstagramMessenger(http_client, get_messenger_settings())
    line = ResultLine(position=1, total=3, name="Kronotrop", area="Beşiktaş", open_now=True)
    result = await messenger.send(
        recipient_id=account.platform_user_id,          # the user's IGSID
        message=result_reply(line, item.source_message_id, wording),
        last_user_message_at=chat_message.received_at,  # Meta's time of their last message
    )
```

- **Replies to saved reels.** A message with `reply_to_message_id` is sent with `reply_to.mid`, so Instagram shows the saved reel above the line.
- **24-hour window.** Before every request, retries included, the client checks the time since the user's last message. Past `INSTAGRAM_REPLY_WINDOW_HOURS` (at most 24; larger values are refused) it sends nothing more: `send` returns `OUTSIDE_REPLY_WINDOW` with the number of requests already made, and logs `dm_skipped_outside_reply_window`. Meta's own "outside the window" answer is reported the same way.
- **Retries.** Only failures that show the message was not taken are retried, with exponential backoff up to `SEND_API_MAX_ATTEMPTS`: Meta's throttling errors and connections that never opened. A timeout after sending or any 5xx, 503 included, may have delivered the message, so it raises `MessengerDeliveryUnknownError` and is not repeated. This avoids repeating a message that may already have been delivered. Meta documents no guarantee about it, so a duplicate is still possible if Meta reports throttling for a message it did take. Every other failure raises a `MessengerError` subclass.
- **Languages.** Every template takes a `ReplyWording`, the bot's own words in the user's language, from `reply_wording(language, localized=None)`:
  - wording passed as `localized` is used as it is, for any language;
  - otherwise the built-in English or Turkish wording, chosen by the primary subtag (`tr-TR` → Turkish);
  - otherwise `ReplyWordingUnavailableError`. Nothing falls back to English silently.

  For other languages the caller supplies the wording. SAV-17's response agent, which already writes replies in the user's language, owns producing it; `has_built_in_wording(language)` tells it when it must. Supplied wording is validated when it is built: every field non-empty, and exactly the placeholders listed on `ReplyWording`, as bare names such as `{item}`. A format spec (`{item:.0}`), a conversion (`{item!r}`), an index or an attribute is refused, so a caller's value is always shown whole. Names, open categories, option labels, count labels and other values are shown exactly as passed in.
- **Logs** hold the kind of message, the attempt, the HTTP status and Meta's error code, never the recipient, a message id, the text or the token.

## Logging

Logs are structured: one JSON object per line, written to stdout. Put data in `extra`, not in the message:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("item_indexed", extra={"item_id": str(item.id), "step": "extract"})
```

Never log secrets, tokens, raw webhook bodies or personal data.

What the logging setup does for you:

- `LOG_LEVEL` applies to every logger, including uvicorn's and other libraries'.
- There is no access log. Uvicorn's access records hold the client's address and the full query string, so they are switched off. To see a request in the logs, log it from the application code, without personal data.
- The API is started with `--ws none`. It has no WebSocket endpoints, and uvicorn would otherwise log the address and query string of every WebSocket handshake.

## Indexing pipeline

Every saved item is understood once, when it is saved (`docs/ARCHITECTURE.md` §3.3, decision record [0012](decisions/0012-indexing-pipeline.md)). The webhook queues one `index_saved_item` job per item, and the worker runs these steps:

1. **Fetch:** download the media from the link the platform sent and keep a 360p copy, a still picture, or a smaller image. A reel shared by DM comes without a file, so it is indexed from its caption only. Nothing is ever read from a post's page.
2. **Classify:** one decision-model call on the media, the caption and the collection hint: the built-in type (or none of them), and whether the post lists several things.
3. **Speech-to-text:** ffprobe decides. A file without an audio track has no speech. A file with one is always transcribed, by the default model and then the fallback. A final transcript with fewer than `TRANSCRIPT_MIN_SPEECH_CHARACTERS` letters and digits counts as no speech; letters of every script count, and so do digits and the marks that combine with letters, while spaces, punctuation and symbols such as ♪ don't. A model that fails is never taken as silence: the job runs again.
4. **Extract:** the text model fills in the type's schema. When a type needs an entity and none was named, the escalation model watches the media.
5. **Normalize:** open values (category, tags, cuisine, subject…) go through the shared vocabulary: exact hit, privacy gate, then match-or-new (§3.4).
6. **Embed** the summary and transcript, mark the item `done`, and **confirm** by DM how it was saved, in the user's language, or in English while it isn't known.

Each step's result is kept in `enrichment_run`, so a job that runs again continues where it stopped. A run that can't read its payload or item, or that passes `INDEXING_RUN_DEADLINE_SECONDS`, runs again too, and an ffmpeg or ffprobe run it was waiting for is stopped and reaped first. That deadline starts before the payload is read, so a slow start (for example, the user's row locked by "delete my data") uses the same budget as the steps. An item gets `INDEXING_MAX_TRIES` runs in all, a hard cap counted on the item across requeues and checked before each run starts; the last marks it `failed` as well as it can. Whatever a job couldn't write, the **stuck-item sweeper** repairs: an arq cron in the worker that, every `INDEXING_SWEEP_INTERVAL_SECONDS`, finds items still `pending` or `enriching` that no job has worked on for `INDEXING_STALE_AFTER_SECONDS`, and queues each again (tries left, payload still there) or marks it `failed`. It reads Postgres only, takes each owner's row lock with `SKIP LOCKED`, and is safe with several workers. The worker reads only the indexing queue; chat jobs wait in theirs. Logs hold, for each item, its type, whether it was escalated, and the cost of its model calls.

Run the worker on your machine (it reads `.env`, and needs ffmpeg, Postgres and Redis):

```bash
docker compose up --detach --wait db redis
python -m app.pipeline.worker
```

Under Compose, the `worker` service runs the same command from the image, which includes ffmpeg. It reads `.env` whole, because it needs nearly every setting, and overrides the database and Redis URLs and the media folder for the Compose network. With the dummy `OPENROUTER_API_KEY` from `.env.example`, every model call is refused, and the items it picks up end as `failed`.

**Media** is kept in `MEDIA_STORAGE_DIR` (`./media` on your machine, git-ignored; the `media_data` volume under Compose), in a folder per user and item. "Delete my data" removes the user's folder. A hosted object store comes with deployment.

**Prompts** are in [`app/pipeline/prompts.py`](../app/pipeline/prompts.py), in one `PipelinePrompts` object with a version. The version is stored with each step's result. Change the version with any change to the wording, and score the change on the eval set (AGENTS.md §4).

The worker reads `Settings`, `ModelSettings`, `MessengerSettings` and `PipelineSettings` (in [`app/config/`](../app/config/), one module per group). The pipeline's own variables:

| Variable | Meaning |
|---|---|
| `MEDIA_STORAGE_DIR` | The folder that stands in for object storage |
| `MEDIA_DOWNLOAD_TIMEOUT_SECONDS`, `MEDIA_MAX_DOWNLOAD_BYTES` | Time limit and largest size of one download. Larger media is skipped and the item indexed from its text |
| `MEDIA_STORED_HEIGHT_PIXELS` | Height of the stored copies (360). Smaller media keeps its size |
| `FFMPEG_PATH`, `FFPROBE_PATH`, `MEDIA_TOOL_TIMEOUT_SECONDS` | The two programs, and the time limit of one run |
| `MEDIA_TOOL_STOP_TIMEOUT_SECONDS` | A run that times out, or is cancelled by the run's deadline, stops its own child: SIGTERM, then SIGKILL if it hasn't exited within this time, then waits up to this time again for it to be reaped. A cancellation during that clean-up doesn't cut it short |
| `TRANSCRIPT_MIN_SPEECH_CHARACTERS` | The near-empty rule above |
| `CLASSIFY_IS_LIST_MIN_PROBABILITY` | A post counts as a list from this probability of yes |
| `EXTRACTION_TEMPERATURE`, `EXTRACTION_MAX_OUTPUT_TOKENS` | Sampling and reply length of extraction and the escalation |
| `VOCAB_PRIVACY_MIN_PROBABILITY` | Wording enters the shared vocabulary only with at least this probability of being generic |
| `VOCAB_ALIAS_MIN_PROBABILITY`, `VOCAB_ALIAS_MIN_USERS` | A candidate alias is verified after matches this confident from this many different users |
| `VOCAB_MAX_VALUE_LENGTH`, `VOCAB_PHONE_MIN_DIGITS` | Longer wording, or wording with a phone-like run of this many digits, never enters the vocabulary |
| `VOCAB_MATCH_CANDIDATES` | How many nearby values the decision model chooses from |
| `INDEXING_CONCURRENCY` | Jobs one worker runs at once; a bulk import fans out over every slot of every worker |
| `INDEXING_MAX_TRIES`, `INDEXING_RETRY_DELAY_SECONDS` | Runs of one job, and the wait before the next run, multiplied by the run's number |
| `INDEXING_RUN_DEADLINE_SECONDS` | How long one run may take, from before it reads the payload to the end of its last step. A run that passes it stops itself, records where it was, and runs again; the last try marks the item `failed` |
| `INDEXING_FAILURE_RECORDING_SECONDS` | How long recording a failure may take after that, from one deadline. If the database doesn't answer in time, the failure is only logged, the job still ends itself, and the sweeper repairs the item later |
| `INDEXING_CLEANUP_RESERVE_SECONDS` | Time kept free for clean-up after a stopped run, at least twice `MEDIA_TOOL_STOP_TIMEOUT_SECONDS` |
| `INDEXING_JOB_TIMEOUT_SECONDS` | arq's own limit on one run, a backstop that cancels without a retry. Must be longer than the run deadline, the recording time and the clean-up reserve together |
| `INDEXING_SWEEP_INTERVAL_SECONDS` | How often the stuck-item sweeper runs. It must divide a minute, or be whole minutes that divide an hour |
| `INDEXING_STALE_AFTER_SECONDS` | How long an unfinished item goes without any job working on it before the sweeper takes it. Longer than a whole run plus every retry wait |
| `INDEXING_SWEEP_BATCH_SIZE` | The most stuck items one sweep repairs. Items whose owner's row is locked are skipped and don't count, so a locked owner can't hold up other users |

## Models

Every model call goes through [OpenRouter](https://openrouter.ai) with one key, and every model sits behind a small interface in [`app/models/base.py`](../app/models/base.py). Which model serves a role is a setting, so a model is swapped in `.env` without a code change. The choices and their measurements are in decision record 0008 (SAV-7); how the clients work is in [0010](decisions/0010-model-clients.md).

| Interface | Role | Model (setting) | Endpoint |
|---|---|---|---|
| `Classifier.decide` | Fixed decisions: typed questions answered with probabilities | `google/gemini-decision-1` (`DECISION_MODEL`) | `POST /api/alpha/decisions`, its own contract, not chat completions |
| `Transcriber.transcribe` | Speech-to-text | `microsoft/mai-transcribe-2` (`TRANSCRIPTION_MODEL`), then `openai/gpt-transcribe` (`TRANSCRIPTION_FALLBACK_MODEL`) | `POST /api/v1/audio/transcriptions` |
| `Generator.generate` | Extraction and the response agent: text, JSON that follows a schema, tool calls | `deepseek/deepseek-v4.1-flash` (`TEXT_MODEL`), pinned to one provider | `POST /api/v1/chat/completions` |
| `Generator.generate` | Escalation: watch the video when the text gave no entity | `google/gemini-2.5-flash` (`ESCALATION_MODEL`) | `POST /api/v1/chat/completions`, with the video inline |
| `Embedder.embed` | Embeddings | `openai/text-embedding-3-large` (`EMBEDDING_MODEL`) at `EMBEDDING_DIMENSION` | `POST /api/v1/embeddings` |

```python
from app.config.models import get_model_settings
from app.models.base import GenerationRequest, Message, YesNoQuestion
from app.models.factory import build_model_clients, open_http_client

settings = get_model_settings()
async with open_http_client(settings) as http_client:
    models = build_model_clients(settings, http_client)

    decision = await models.classifier.decide(
        ["the caption", video], [YesNoQuestion("is_list", "Is this a list of several things?")]
    )
    transcript = await models.transcriber.transcribe(audio)
    reply = await models.text_generator.generate(
        GenerationRequest(messages=[Message(role="user", content="…")])
    )
    vectors = await models.embedder.embed(["a cosy café"])
```

**The boundary.** Code outside `app/models/` imports only `app.models.base` (the interfaces), `app.models.errors` and `app.models.factory`. It never imports a model SDK, a client class or the HTTP transport. `tests/models/test_model_boundary.py` fails if it does.

**In tests, use the fakes** in [`app/models/fakes.py`](../app/models/fakes.py): `FakeClassifier`, `FakeTranscriber`, `FakeGenerator` and `FakeEmbedder`. They make no network call, return what the test scripted (or raise a scripted error), and record the calls they received.

**What the results tell you**

- **Decisions:** one answer per question, by key. A question the model refused, left out or answered unusably comes back as `Unanswered`; the others stay valid.
- **Transcripts:** `model` is the model whose answer it is, and `attempts` lists every model that was asked. `language`, `segments` and `words` are `None` when the model did not return them, which is always the case for the fallback model. **An empty transcript is a result, not an error:** it means no model heard words. A **failure** is raised as `TranscriptionFailedError`, and only when both models failed. The full table is in [record 0010](decisions/0010-model-clients.md#speech-to-text-an-empty-transcript-is-a-result-a-failure-is-an-error).
- **Generation:** `structured` holds the parsed JSON when a schema was requested. `tool_calls` holds the calls with parsed arguments.
- **Embeddings:** every vector has exactly `EMBEDDING_DIMENSION` components, or the call raises `EmbeddingDimensionError`.

**Errors** are all subclasses of `ModelError` ([`app/models/errors.py`](../app/models/errors.py)). A call that times out, is rate-limited or hits a server error is retried with a doubling wait (`OPENROUTER_MAX_ATTEMPTS`, `OPENROUTER_RETRY_*`); anything else fails at once.

| Error | Meaning |
|---|---|
| `ModelRequestError` | The request was rejected. Subclasses: `ModelAuthenticationError` (key refused), `ModelCreditsError` (out of credits) |
| `ModelRateLimitError`, `ModelUnavailableError` | Still rate-limited, or timeouts and server errors, after the last attempt |
| `ProviderRouteUnavailableError` | The pinned provider cannot serve the call, and no other provider is allowed |
| `ProviderPinViolationError` | The reply does not prove that the pinned provider, and only it, served it. The reply is discarded |
| `ModelResponseError` | The answer was unusable. Subclasses: `SchemaOutputError`, `EmbeddingDimensionError` |
| `TranscriptionFailedError` | Both speech-to-text models failed |

**Nothing private is logged.** Logs and error messages hold the operation, the model, the HTTP status and the attempt number. They never hold the key, a caption, a transcript, a message, media, or a provider's own error text.

### The DeepSeek provider pin

DeepSeek's own API is hosted in China, and OpenRouter by default sends a model to whichever provider it likes. So the text model is **pinned to one provider endpoint hosted outside China**, and the pin is a setting:

```bash
TEXT_MODEL_PROVIDER=coreweave/fp8     # the endpoint's OpenRouter slug
TEXT_MODEL_PROVIDER_NAME=CoreWeave    # the name OpenRouter reports for it in routing metadata
```

- **Every request allows only that endpoint:** `provider: {"only": [slug], "order": [slug], "allow_fallbacks": false}`.
- **Every reply must prove where it was served.** Each request sends `X-OpenRouter-Metadata: enabled`, and OpenRouter then adds `openrouter_metadata` to the reply. The reply is used only if exactly one endpoint is `selected`, it is the pinned provider with the configured model, and every attempt the router made went to that provider. After more than one attempt the history must be complete (one entry per attempt, the last one successful), or the earlier attempts are unidentified. Anything else (no metadata, malformed or ambiguous metadata, a missing or invalid attempt number, missing or incomplete history, another provider, a top-level `provider` field that disagrees) discards the reply with `ProviderPinViolationError`. Another provider listed only as an unselected candidate is fine: it was considered, not called.
- **Cache replays are refused.** OpenRouter strips the metadata from them, so they can't be proven. Keep response caching off for the text model.
- **No route means an error, never another host.** If the endpoint cannot serve a call (OpenRouter answers 404 or 503; a 503 is retried first), `generate` raises `ProviderRouteUnavailableError`.

**Why CoreWeave** (evidence read on 17 Nov 2026; the full comparison is in [record 0010](decisions/0010-model-clients.md#the-provider-that-serves-deepseek)):

- OpenRouter's provider list gives CoreWeave `datacenters: ["US"]`.
- CoreWeave's own list of all its availability zones has 60 zones in the United States, Canada and Europe, and none in China or elsewhere in Asia.
- CoreWeave lists DeepSeek V4.1 Flash among its hosted models, and OpenRouter's endpoint supports JSON-schema output and tool calls.

No page says which zone serves the endpoint, and **no live call has confirmed the route yet**. The choice is waiting for the owner's confirmation. To change the provider, change both variables and add the new provider's serving-region evidence to a decision record. A company's home country alone is not evidence.

### Model settings

The model clients read these variables through `ModelSettings` in [`app/config/models.py`](../app/config/models.py), with `get_model_settings()`. It is a separate class from the API's `Settings`: the API and the migrations call no model, so they start without these. The same rules apply: no variable has a default, and an error names the variable without printing its value. The values in `.env.example` are the approved models; only `OPENROUTER_API_KEY` needs a real value, and only for real model calls.

| Variable | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | The one key for every model call. Never commit a real one |
| `OPENROUTER_BASE_URL` | Base URL for chat completions, transcriptions and embeddings: `https://openrouter.ai/api/v1`. Must be `https` |
| `OPENROUTER_DECISIONS_URL` | Full URL of the decisions endpoint, which is not chat completions: `https://openrouter.ai/api/alpha/decisions`. Must be `https` |
| `OPENROUTER_MAX_ATTEMPTS` | How often one call is tried in total when it times out, is rate-limited or hits a server error. At least 1 |
| `OPENROUTER_RETRY_BASE_DELAY_SECONDS`, `OPENROUTER_RETRY_MAX_DELAY_SECONDS` | The wait before the second try, which doubles each time, and the longest wait |
| `DECISION_MODEL`, `DECISION_TIMEOUT_SECONDS` | Decision model id (`google/gemini-decision-1`) and the time limit of one request |
| `TEXT_MODEL`, `TEXT_TIMEOUT_SECONDS` | Text model id (`deepseek/deepseek-v4.1-flash`) and the time limit of one request |
| `TEXT_MODEL_PROVIDER` | The one provider endpoint that may serve the text model, as an OpenRouter slug (`coreweave/fp8`). Must be hosted outside China. See [The DeepSeek provider pin](#the-deepseek-provider-pin) |
| `TEXT_MODEL_PROVIDER_NAME` | The name OpenRouter reports for that provider in a reply's routing metadata (`CoreWeave`). A reply not proven to come from it is discarded |
| `ESCALATION_MODEL`, `ESCALATION_TIMEOUT_SECONDS` | Video model id (`google/gemini-2.5-flash`) and the time limit of one request |
| `TRANSCRIPTION_MODEL`, `TRANSCRIPTION_RESPONSE_FORMAT` | Default speech-to-text model (`microsoft/mai-transcribe-2`) and its response format: `verbose_json` (with language and timestamps) or `json` |
| `TRANSCRIPTION_FALLBACK_MODEL`, `TRANSCRIPTION_FALLBACK_RESPONSE_FORMAT` | The model asked when the default fails or returns no words (`openai/gpt-transcribe`, which only accepts `json`) |
| `TRANSCRIPTION_TIMEOUT_SECONDS` | Time limit of one transcription request, for either model |
| `EMBEDDING_MODEL`, `EMBEDDING_TIMEOUT_SECONDS` | Embedding model id (`openai/text-embedding-3-large`) and the time limit of one request. The vector size is `EMBEDDING_DIMENSION` |
| `EMBEDDING_DIMENSION` | Shared with the database (see [Configuration](#configuration)): the vector size the embedding model is asked for, and the size every returned vector must have |

The Compose `api` service does not receive these variables, because the API calls no model. The `worker` service reads them from `.env` (see [Indexing pipeline](#indexing-pipeline)).

### Live smoke tests

`tests/live/` holds one real call per model client. They are **skipped** unless both `LIVE_MODEL_TESTS=1` and `OPENROUTER_API_KEY` are set, so `make check` never spends money. They send only invented text and one second of generated silence, and cost a fraction of a cent together. **A live run needs the owner's approval first.**

Tests do not read `.env`, so load it into your shell first:

```bash
set -a; source .env; set +a
LIVE_MODEL_TESTS=1 pytest -m live
```

The text-model test also checks, from the reply's routing metadata, that the pinned provider served it. Sending a video to the escalation model is not covered: the test sends text only.

## Content types

Every saved post gets an open **category** and **tags**, written by the text model in plain words and in any language ("restaurant", "car review", "ギターレッスン"). Nothing is picked from a list in code, and nothing is tied to a city, a country or a language (decision record 0005, SAV-26).

A **built-in type** exists only where a kind of post gets extra behaviour: more fields, a resolver or more search filters. Each one is a module in `app/types/` that declares a `ContentType`, and [`app/types/registry.py`](../app/types/registry.py) lists them. The rest of the code asks the registry and reads the declaration. It never checks for a type by name, and a test (`tests/types/test_content_type_registry.py`) fails if it does.

| Built-in type | Fields added to the schema | Resolver | Extra search filters |
|---|---|---|---|
| `place` | per venue: `name`, `area_hint`, `cuisine` | `google_places` | `area`, `near`, `open_now`, `cuisine`, `price` |
| `movie` (also series) | per title: `title`, `year_hint` | `tmdb` | `genre`, `year`, `streaming`, `watched` |
| `study` | `subject`, `topic`, `format`, `key_points`, `resources`, `exam` | none | `subject`, `topic`, `format`, `exam` |
| *(no built-in type)* | none: `summary`, `category` and `tags` only | none | none |

A post that fits no built-in type is not an error and is not pushed into the nearest one. The decision model answers "none of these", the post has no type, and it keeps its summary, category and tags:

```python
from app.types.registry import CONTENT_TYPES

place = CONTENT_TYPES.get("place")
place.extraction_schema.model_validate(text_model_output)       # raises on malformed output
[search_filter.name for search_filter in place.search_filters]

CONTENT_TYPES.extraction_schema("place")   # PlaceExtraction
CONTENT_TYPES.extraction_schema(None)      # Extraction: summary, category, tags
```

A declaration has these parts (all defined in [`app/types/base.py`](../app/types/base.py)):

| Part | What it is |
|---|---|
| `name`, `description` | The name stored on a saved item, and the text the decision model sees when it picks a post's type |
| `extraction_schema` | The Pydantic model the text model fills in. It extends `Extraction`, so it always has `summary`, `category` and `tags`. Unknown keys, missing keys, blank text and wrong shapes are rejected. The values themselves are open |
| `resolver` | The name of the resolver in `app/resolvers/` that looks up the type's entities, or `None` |
| `search_filters` | The filters the type adds to search. `category` and `tags` work for every item and are not declared per type |

**Open values and the vocabulary.** A field such as `cuisine` or `subject` is open text. To keep "korean", "Korean food" and "Kore mutfağı" from becoming three values, the field is marked with `Vocabulary("cuisine")`. The normalization step (a later ticket) reads the marks with `vocabulary_fields(...)` and matches each new value against the shared vocabulary. A filter of kind `value` names the same vocabulary field, so a question in one language finds a value saved in another.

| Filter kind | Takes |
|---|---|
| `value` | An open value, normalized to one that exists in the user's own saves |
| `area` | A place name at any level, anywhere: country, region, city, district or neighbourhood |
| `location` | A point and a distance around it |
| `boolean`, `integer` | True or false, or a whole number |
| `text` | Free text that is matched as it is |

### How to add a new type

Add a type only when it needs something that the open category can't give: its own fields, a resolver or its own filters. Example: a `recipe` type.

1. **Create `app/types/recipe.py`.** Copy `study.py` for a type without a resolver, or `movie.py` for one with a resolver.
2. **Write the extraction schema.** Extend `Extraction` (it already has `summary`, `category` and `tags`) and give every field a `description`, because the text model reads it. If a post points at real-world things that a resolver looks up, extend `EntityExtraction` instead and put them in `entities`, with a mention class that extends `EntityMention`.
3. **Keep values open.** Never add an enum or a list of real-world values (diets, cuisines, cities…). Use `NonEmptyText`, and mark a field whose values should be kept consistent: `diet: Annotated[NonEmptyText | None, Vocabulary("diet")]`. Write examples in the description, not rules, and don't assume a language or a country.
4. **Declare the `ContentType`:** the name, a one-line description, the resolver name (or `None`) and the search filters. Give a `value` filter the same vocabulary field as the schema field it searches.
5. **Register it:** add it to `CONTENT_TYPES` in `app/types/registry.py`. This is the only existing file you change.
6. **Test it:** in `tests/types/test_content_type_schemas.py`, add a good example, values nobody listed in several languages, and malformed output. In `tests/types/test_content_type_registry.py`, add the expected resolver, filters and vocabulary fields.
7. **Update the docs:** the table above and `docs/ARCHITECTURE.md` §3.4.

A filter also declares its **source**, where search reads what it is checked against (`FilterSource`):

| Source | Search reads | Used by |
|---|---|---|
| `saved_values` (the default) | The key named after the filter: in the item's `fields` when the type's schema has it as a top-level field (study `subject`), otherwise in the linked entity's `data` or in that entity's own mention in `fields.entities` (place `cuisine`); for `area` the entities' address levels, for `location` their coordinates | most filters |
| `country_availability` | `entity_availability` for the asking user's country, offer kinds `flatrate`, `free` and `ads` only. Only for `value` filters | `streaming` |
| `user_status` | Whether the asking user marked the item's entity as done. Only for `boolean` filters | `watched` |
| `opening_hours` | The entity's opening hours. Only for `boolean` filters. Search rejects it until the place resolver stores hours | `open_now` |

A declaration that can't work fails when the module is imported, with a `ContentTypeDefinitionError`: a repeated filter or type name, a vocabulary field on a filter that is not a `value`, a source that cannot serve the filter's kind, or a resolver on a type whose schema has no `entities` (or the reverse).

## Search

[`app/agent/search/retrieval.py`](../app/agent/search/retrieval.py) searches one user's saves (ARCHITECTURE §3.5 and §3.6, decision record 0014). The response agent's tools (SAV-17) build a `SearchQuery` and call `HybridRetriever.search()`:

```python
result = await retriever.search(session, SearchQuery(
    user_id=user.id, country="US", type="movie",
    filters={"genre": "horror", "streaming": "netflx"},
))
result.hits                # item ids, best first, each with the level that found it
result.level               # L0, L1 or L2
result.normalized          # "netflx" → "Netflix", and how
result.availability_gaps   # (movie, country) rows to fetch or refresh (SAV-15)
```

1. **Check** the query against the content types: an unknown type or filter, or a value of the wrong kind, raises `SearchQueryError`. A streaming filter without the user's country raises `CountryRequiredError`, so the agent asks once.
2. **Normalize** what the user wrote to a value in **their own** saves: an exact hit (their value, or a canonical value or verified alias in the vocabulary) needs no model call; otherwise Gemini Decision 1 picks one of their values or "none of these". No other user's values are ever read, and search never writes to the vocabulary.
3. **L0:** SQL/PostGIS filters, then pgvector ranking by the query text. The type's filters must all hold on the same place or title of a list post. Without a text, places near a point come nearest first, and everything else newest first.
4. **L1:** if the embedding or the vector search fails, the same filters plus Postgres full-text search on the caption, summary and transcript.
5. **L2:** with fewer than `SEARCH_MIN_RESULTS` results, a district or neighbourhood widens to its city, tags and the type's other filters are dropped (a keyword match from L1 is kept), and items that are not fully indexed yet are searched by the words of the query. The result lists what was relaxed.

Every step up the ladder is logged as `search_fallback` with its level and reason, never with the query's text. The whole search runs within `SEARCH_TIME_BUDGET_SECONDS`; when it runs out, the result has what was found and `incomplete=True`.

The wording sent to the decision model is `RETRIEVAL_PROMPTS` in [`app/agent/search/prompts.py`](../app/agent/search/prompts.py), version `2026-11-18.1`, approved by the owner. Any change to it bumps the version and is scored on the evaluation set.

A streaming filter matches a title only when it's watchable on that platform in the user's country without paying extra: listed under the offer kind `flatrate`, `free` or `ads` in `entity_availability.providers` (`{"<offer kind>": ["<provider name>", …]}`). `rent` and `buy` offers are stored for replies but never match.

| Variable | Meaning |
|---|---|
| `SEARCH_TIME_BUDGET_SECONDS` | How long one search may take in total, model calls included |
| `SEARCH_MAX_RESULTS` | The most items one search returns |
| `SEARCH_MIN_RESULTS` | Fewer results than this climb to L2. At most `SEARCH_MAX_RESULTS` |
| `FILTER_NORMALIZATION_MIN_PROBABILITY` | The decision model's lowest probability for a value that counts as what the user meant, 0 to 1 |
| `FILTER_NORMALIZATION_MAX_OPTIONS` | The most saved values offered to the decision model for one filter |
| `AVAILABILITY_MAX_AGE_DAYS` | Streaming availability older than this is reported as stale for that country |

They are read through `RetrievalSettings` in [`app/config/retrieval.py`](../app/config/retrieval.py), with `get_retrieval_settings()`, with the same rules as the other settings classes.

### Filter normalization evals

[`scripts/run_normalization_evals.py`](../scripts/run_normalization_evals.py) measures how often normalization picks the value the user meant (SAV-30). [`evals/normalization_cases.jsonl`](../evals/normalization_cases.jsonl) holds about 100 synthetic cases, one per line: the filter slot, one user's saved values for it (near-misses included on purpose), what the user wrote, and the value they meant, or `null` when the right answer is "none of these". Each case also has a kind (`exact`, `typo`, `synonym`, `translation`, `granularity`, `no_match`, `large_list`) and the wording's language. A case may list `verified_aliases` (alias → saved value) that the shared vocabulary already holds.

The runner creates a throwaway database `savely_eval_<random>` on the Postgres server of `DATABASE_URL`, migrates it, and drops it at the end; it never touches the tables of the database you develop against. Each case gets its own user, and its rows are rolled back before the next case. Normalization runs with the configured `FILTER_NORMALIZATION_MIN_PROBABILITY`, `FILTER_NORMALIZATION_MAX_OPTIONS` and prompt version, unchanged. The markdown report goes to standard output, ready to paste into a ticket:

- correct, **wrong pick** (a different value was used), **missed** (a value was expected and the wording was kept) and **false match** (none was expected and a value was used), overall, per kind and per language;
- **shortlist recall**: how often the expected value was among the options shown to the decision model;
- how each case was resolved (`saved_value`, `vocabulary`, `decision`, `unchanged`) and how often the model was called;
- a **threshold sweep** at 0.5–0.9, computed from the probabilities the model returned, so it needs no extra calls;
- latency per case, and the cost the provider reported;
- every case that was not correct.

Run it from the repository root with `db` running and the settings in `.env` or the environment:

```bash
docker compose up --detach --wait db

# Offline: fake models, no key, no cost. The decision model never answers, so this
# checks the setup and shows what exact hits alone get right.
python -m scripts.run_normalization_evals

# Real models through OpenRouter: reads OPENROUTER_API_KEY and the model settings.
python -m scripts.run_normalization_evals --live > normalization-report.md
```

A live run makes at most one decision call and one embedding call per case, plus a few calls to embed the vocabulary values first: well under $1 for the whole set. It needs the owner's approval like every live call. `--cases <file>` runs another case file, and `--database-url <url>` points at another Postgres server (any database on it; only `CREATE DATABASE` and `DROP DATABASE` run there).

## Response agent

The response agent answers a user's DM about their saves ([`ARCHITECTURE.md`](ARCHITECTURE.md) §3.5, decision record [0015](decisions/0015-response-agent.md)). It never queries the database: the text model (DeepSeek V4.1 Flash) calls the tools in [`app/agent/tools.py`](../app/agent/tools.py), and the tools search and write.

In production the chat worker builds and runs it (see [Chat worker](#chat-worker)); the core of one turn is:

```python
toolbox = build_toolbox(sessions, redis, classifier=models.classifier, embedder=models.embedder,
                        settings=settings, pipeline_settings=pipeline_settings,
                        chat_settings=chat_settings)  # app/agent/worker.py
agent = ResponseAgent(generator=models.text_generator, toolbox=toolbox,
                      prompts=AGENT_PROMPTS, settings=get_agent_settings())
answer = await agent.answer(AgentTurn(user_id=sender_id, country=user.country,
                                      text="only the cheap ones", history=history,
                                      working_set=last_ids, working_set_shown=5,
                                      working_set_searches=last_searches,
                                      home_area=user.home_area))
await send_answer(AnswerSender(messenger), toolbox, sender_id, igsid, answer,
                  last_user_message_at)
```

| Tool | What it does |
|---|---|
| `search_items` | Hybrid search: collection, type, category, tags, the type's filters, text, platform, date |
| `near` | Search around a point, for the types that have a location filter (places) |
| `get_item` | One item's details: open values with their paths, entities, links, streaming in the user's country |
| `update_item` | Correct one open value of one item. Only that item changes; no synonym is created |
| `set_status` | Mark the place or title behind an item as done (visited, watched, read) or wanted; an item without one (e.g. a study post) keeps the status on itself (`saved_item.user_status`) |
| `set_profile` | Keeps where the user lives (`user.home_area`) and their country (`user.country`), when they tell it |
| `request_data_deletion` | "Delete my data" in any words or language: the bot asks a fixed confirmation question |
| `confirm_data_deletion` | Deletes everything of the sender, only on an explicit confirmation of a delivered question, in time |

- **Ownership comes from the DM.** `Toolbox.for_user` binds the tools to the sender's `user_id` and country. No tool takes a user id, and another user's item is answered like a missing one.
- **The answer is checked** before it is sent: it may only list items the tools returned or the user's last results, it must carry a notice when a search fell back (`backup_mode`, `broader_results`, `may_be_incomplete`), and it must fit in one DM. A wrong answer goes back to the model, within `AGENT_MAX_STEPS`; the last call may not call tools.
- **Any language.** The model replies in the language of the user's message and names it. For a language without built-in reply wording, the agent asks the model once for the bot's fixed words in it; unusable wording is an error, never another language.
- **Sending** (`app/agent/replies/answers.py`): an intro, then one reply per result that quotes the saved reel, at most `AGENT_MAX_RESULTS_PER_ANSWER`. A refused quote falls back to the post link (a preview card), then to the name with its Maps or TMDB link (decision record 0004). Meta answers an accepted reply the same way whether or not Instagram shows the quoted reel, so only a refused reply falls back (record 0015, Q7).
- **"Delete my data" takes two steps, enforced in code; the model can never delete on its own word.** `request_data_deletion` makes the answer a fixed question in the user's language, with one **"Yes, delete" button** (a quick reply with the fixed payload `DELETION_CONFIRM_PAYLOAD`): what is deleted (saves, stored media, the conversation, the profile), that it can't be undone, and to tap the button. The fixed "done" message is prepared in the same language at the same time. `send_answer` (`app/agent/send.py`) stores the request in Redis (`app/agent/deletion_requests.py`) only after the whole answer was sent, with its deadline, `DATA_DELETION_CONFIRM_SECONDS` later. The request is taken out at the start of the user's next turn, so "no" or any other message cancels it. The Instagram adapter keeps a tapped button's payload as `ChatMessage.quick_reply_payload`; the chat worker (`app/agent/chat.py`) sets `AgentTurn.deletion_confirmed` only when `confirms_deletion(payload)` holds. `confirm_data_deletion` erases only with that confirmation and before the deadline, checked right before erasing and again by the eraser once it holds the user's row lock, which it may have to wait for; a typed "yes" shows the question and its button again and deletes nothing. It then runs `UserDataEraser` (`app/db/deletion.py`): ingestion's Redis state and jobs, the pending request, short-term memory and the stored media first, then the Postgres rows (the profile and `conversation_msg` go with the `user` row). A failure outside Postgres leaves Postgres unchanged, and a repeat succeeds. A successful erasure ends the turn at once with the prepared "done" message, sent to `AgentAnswer.deleted_recipients` (read before the deletion); no model or tool runs after it. The log lines `data_deletion_requested`, `data_deletion_confirmation_delivered` and `user_data_deleted` hold only the event and the number of items, and the agent's own lines (`agent_tool_called`, `agent_answered`) hold no user id; `tests/agent/test_data_deletion.py` checks that no id or message text is logged during a deletion.
- **Corrections** (`app/pipeline/vocabulary/corrections.py`) go through the vocabulary's normal steps, keep the user's wording marked as set by hand, and count against the alias that produced the wrong value (`correction_count`, and `corrected_alias_ids` on the corrected path's normalize-step provenance). A corrected candidate alias is never verified. A verified alias that `VOCAB_ALIAS_REVIEW_MIN_USERS` different users corrected is logged as `vocabulary_alias_flagged_for_review`; one user's repeated corrections count once (record 0015, Q4).

The wording sent to the text model is `AGENT_PROMPTS` in [`app/agent/prompts.py`](../app/agent/prompts.py), version `2026-11-21.1`: the owner approved `2026-11-20.3` for now (record 0015, Q3), and SAV-18 added the profile, the follow-up rules and the working set's searches (record 0016). The tests use a fake model, so they check the loop and the tools, not the model's quality; the wording is scored on the evaluation set when SAV-19/20 exist.

| Variable | Meaning |
|---|---|
| `AGENT_MAX_STEPS` | The most model calls one answer may take, tool rounds included; at least 2 |
| `AGENT_MAX_RESULTS_PER_ANSWER` | The most saved items one answer sends back; the rest are kept for "show more" |
| `AGENT_MAX_OUTPUT_TOKENS` | The longest reply the text model may write per call |
| `VOCAB_ALIAS_REVIEW_MIN_USERS` | Different users whose corrections flag a verified vocabulary alias for review; at least 2 |
| `DATA_DELETION_CONFIRM_SECONDS` | How long a delivered "delete my data" question may be confirmed |

They are read through `AgentSettings` in [`app/config/agent.py`](../app/config/agent.py), with `get_agent_settings()`, with the same rules as the other settings classes.

## Chat worker

The chat worker answers the texts users write to the bot (decision records [0015](decisions/0015-response-agent.md) Q5 and [0016](decisions/0016-short-term-memory-and-chat-worker.md)). It runs `answer_chat_message` jobs from the chat queue (`arq:queue:chat`), separate from indexing:

```bash
python -m app.agent.worker
```

Under Compose, the `chat_worker` service runs the same command. Like the indexing worker it reads `.env` whole, and it shares the media volume, because "delete my data" from a DM also erases stored media. It reads `Settings`, `ModelSettings`, `PipelineSettings`, `RetrievalSettings`, `AgentSettings`, `MessengerSettings` and `ChatSettings`.

For each text ([`app/agent/chat.py`](../app/agent/chat.py)):

1. Read the message from ingestion's payloads; none means the user was deleted, and the job stops.
2. Load the sender's profile (their IGSID, country and home area) and short-term memory (the last messages and the working set).
3. Run the response agent for that user only, with the user taken from the DM's sender. `deletion_confirmed` is set only when the message is a tap on the "Yes, delete" button; the tools still require that it is the sender's next message after a delivered question.
4. Send the answer within Instagram's 24-hour window, counted from when the user sent the text.
5. Store the exchange and the new working set in Redis, the messages in `conversation_msg`, and the language the user wrote in as `user.language` (the save confirmations use it). Nothing is stored after a deletion.

**Short-term memory** ([`app/memory/session.py`](../app/memory/session.py)) keeps three keys per user, `savely:memory:<user id>:history|working_set|replies`, for `SESSION_TTL_SECONDS` after the last exchange. The history is the last `SESSION_HISTORY_MESSAGES` messages. The working set is the last answer's results in their numbered order, how many were shown, and the searches (tool and arguments) they came from: the model sees them by number for "the second one", knows which were not shown for "show more", and repeats the search with one more filter for "only the cheap ones". An answer that lists no items keeps the last working set. `replies` marks the messages whose answer started to be sent, so a retried job never sends twice.

**The profile** is the `user` row: `set_profile` keeps `home_area` and `country` when the user tells them ("I live in Beşiktaş"), and the model asks once when a question needs one it doesn't know. The home area is shown to the model on every turn.

**Retries.** Preparing an answer has `CHAT_RUN_DEADLINE_SECONDS`. A failure there that may pass (a model outage, the deadline, Redis or Postgres) runs the job again after `CHAT_RETRY_DELAY_SECONDS` × the try's number, up to `CHAT_MAX_TRIES`; the last try sends the bot's error message in the user's language instead. A failure that would repeat (a rejected request, the step limit) sends the error message at once. A send that fails is logged (`chat_answer_send_failed`) and not tried again. A "Yes, delete" tap whose turn fails deletes nothing, because the request was used up: the user asks again.

**"Delete my data"** goes through the one `UserDataEraser` that `build_toolbox` builds with every store outside Postgres: ingestion's Redis state and jobs, the pending deletion request, short-term memory and the media store. The profile fields and `conversation_msg` rows are deleted with the `user` row.

| Variable | Meaning |
|---|---|
| `SESSION_TTL_SECONDS` | How long short-term memory lives after the last exchange (about 24 hours) |
| `SESSION_HISTORY_MESSAGES` | How many of the last messages the response agent reads (about 20) |
| `CHAT_CONCURRENCY` | Messages one chat worker answers at once |
| `CHAT_MAX_TRIES` | Runs of one message's job before the error message is sent |
| `CHAT_RETRY_DELAY_SECONDS` | The wait before a failed job runs again, times the try's number |
| `CHAT_RUN_DEADLINE_SECONDS` | How long one try may take to prepare an answer; sending is not cut off |
| `CHAT_JOB_TIMEOUT_SECONDS` | arq's own limit on one try; longer than the run deadline, only a backstop |

They are read through `ChatSettings` in [`app/config/chat.py`](../app/config/chat.py), with `get_chat_settings()`. The log lines (`chat_answer_retry`, `chat_answer_failed`, `chat_answer_send_failed`, `user_profile_updated`) hold error class names and field names only, never a message, a place or an id.

## Project layout

```
app/
  main.py             # FastAPI app factory and GET /health
  config/             # settings from environment variables, one module per group:
                      # base (the API's Settings), models, messenger, pipeline, retrieval, agent, chat
  logging_config.py   # JSON logging setup
  adapters/
    base.py           # CaptureEvent and ChatMessage: what every adapter outputs
    instagram.py      # signature check, webhook parsing → CaptureEvent / ChatMessage
  api/
    webhooks.py       # GET and POST /webhooks/instagram
  ingestion/
    service.py        # stores captures, routes texts, queues jobs
    repository.py     # users on first contact, saved items, collection hints
    jobs.py           # job names, ids and queues (one arq queue per job name)
    session_state.py  # per-user state in Redis: payloads, text receipts, recent shares
    erasure.py        # erases a user's Redis state and jobs for "delete my data"
  pipeline/           # indexing: the arq worker, indexer and prompts at the top (see "Indexing pipeline")
    steps/            # fetch, classify, transcribe, extract, embed, confirm
    media/            # ffprobe/ffmpeg tools and the media store
    vocabulary/       # open values, normalization and the shared vocabulary
    recovery/         # per-item run state and the stuck-item sweeper
  types/              # content-type registry: one module per built-in type
  models/             # model interfaces, errors, factory and fakes (see "Models")
    openrouter/       # the shared HTTP transport, routing checks, JSON helpers
    clients/          # one client per model role
  agent/
    agent.py          # the response agent's tool-calling loop and answer checks
    tools.py          # search_items, near, get_item, update_item, set_status, set_profile, deletion tools
    chat.py           # answers one chat message: memory in, agent, send, memory out
    worker.py         # the chat worker (arq) and the eraser of every store
    send.py           # sends an answer; a delivered deletion question becomes confirmable
    prompts.py        # versioned wording for the text model
    deletion_requests.py # pending "delete my data" requests, with their deadline
    search/
      retrieval.py    # hybrid search: normalize, filter, rank, fallback L1–L2, time budget
      query.py        # SearchQuery, checked against the content types
      conditions.py   # SQL/PostGIS/pgvector conditions and rankings
      saved_values.py # a user's own values for a filter, ranked by meaning among them
      normalization.py # "netflx" → Netflix: exact hit, then the decision model
      fallback.py     # fallback levels, reasons and their log line
      prompts.py      # versioned wording for the decision model
    replies/
      messenger.py    # Instagram Send API client: DMs, quick replies, replies to saved reels
      templates.py    # fixed DM wording in the user's language
      answers.py      # an answer: the intro, then one reply per result, with fallbacks
  memory/
    session.py        # short-term memory in Redis: history, working set, reply marks
    conversation.py   # the sender's profile and `conversation_msg` in Postgres
  db/
    models.py         # SQLAlchemy models
    session.py        # the API's engine and session factory
    types.py          # column types for pgvector and PostGIS
    deletion.py       # "delete my data" for one user
    migrations/       # Alembic environment and revisions
alembic.ini           # Alembic configuration
scripts/spikes/       # reproducible experiments behind decision records (see its README)
tests/                # pytest suite, in folders that mirror app/
docker/postgres/      # Postgres 16 image with PostGIS and pgvector
docker-compose.yml    # local stack: db, redis, api, worker, chat_worker
Dockerfile            # the image of the API and the workers
Makefile              # install, fmt, lint, typecheck, test, check, run, migrate, up, down
README.md             # product front page for visitors
assets/               # the logo
docs/
  ARCHITECTURE.md     # the design (source of truth)
  development.md      # this guide
  decisions/          # decision records
```

The full target layout is in [`docs/ARCHITECTURE.md` §6](ARCHITECTURE.md#6-repository-layout). Folders are added when a ticket needs them.
