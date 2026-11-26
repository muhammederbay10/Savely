# 0002: What Instagram's `messages` webhooks really contain

- Status: findings recorded. Points 1–3 under [Consequences](#consequences) need Muhammed's decision before SAV-11 and SAV-14 build on them.
- Date: 2026-11-16
- Superseded in part: the time-only collection hint (the last adapter point of consequence 4) is replaced by record [0018](0018-model-decides-collection-hints.md) (SAV-39): the window is only an upper bound, and the decision model decides.
- Ticket: SAV-5 (the ticket asked for `0001-…`; that number was taken by the SAV-4 scaffold record, so this is `0002`)

## Context

`docs/ARCHITECTURE.md` §3.1 describes the webhook from Meta's documentation, and Meta is replacing the old `share` attachment with `ig_post` during 2026. Before writing the Instagram adapter (SAV-11), we recorded what the bot `@savedagent2026` actually receives. Muhammed sent DMs from his personal account (an Instagram Tester) while the app was in development mode. A throwaway recorder behind an ngrok tunnel checked each signature and saved the raw bodies outside the repo. The anonymized results are in `tests/fixtures/webhooks/`.

## Findings

### Shape
1. **Real DMs use `entry[].messaging[]`.** Each webhook held one entry with one messaging event: `sender.id`, `recipient.id`, `timestamp`, and `message` with `mid` plus either `text` or `attachments`.
2. **The dashboard's Test button sends a different shape.** It uses `entry[].changes[]` with `field: "messages"` and a nested `value`. The adapter must be built and tested against the real shape, not the test sample.
3. **`entry.id` and `recipient.id` are the bot's Instagram account id.** `sender.id` is a 16-digit, app-scoped id for the sender, not their username.
4. **Times are in milliseconds.** `entry.time` came 1.4–2.9 s after `messaging.timestamp`.
5. **Every body was signed.** `X-Hub-Signature-256` matched an HMAC-SHA256 of the raw body with the **Meta App Secret** (App settings → Basic). The separate Instagram app secret wasn't needed.

### Attachments
6. **Reels arrive as `ig_reel`** with `reel_video_id`, `title` and `url`.
   - `title` is the reel's full caption: a 490-character Turkish caption came through whole, with its line breaks, hashtag and emoji.
   - **`url` is the reel's public page (`https://www.instagram.com/reel/<shortcode>/`), not a video file.** Fetching it returns an HTML page.
7. **Posts arrive as `ig_post`.** No `share` attachment appeared. The payload has `ig_post_media_id`, `url` and, when the post has a caption, `title`.
   - `url` is a signed CDN link (`lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=<ig_post_media_id>&signature=…`). It returned the JPEG directly, about 2 minutes after sending. The link has no expiry field, and the response says `no-store`.
8. **The carousel arrived as a single `ig_post` with one image URL and no `title`.** The other items weren't included. Muhammed confirmed afterwards that the post he sent **had a caption**, so the caption was dropped, not missing.
9. **A share and the text after it are two separate webhooks.** The reel and "For my movies collection" arrived 9.1 s apart, each as its own event, with nothing linking them except the sender and the timing.

### Setup that was needed
10. **No DM arrived until the bot's access token had been generated and the webhook subscription switched off and on again.** Before that, the dashboard Test reached us, but real DMs didn't. Accepting the chat in the bot's Requests folder didn't change anything on its own.
11. **The first chat landed in the bot's Requests folder**, because the bot doesn't follow the sender. Once the token step was done, messages from that chat were delivered. Whether a brand-new user's *first* message is delivered without anyone accepting it hasn't been tested yet; that needs a second tester who never chatted with the bot.

## Consequences

1. **Reels: we don't get the video file (decision needed).** The pipeline in §3.3 starts with Gemini Decision 1 on the video, and speech-to-text on its audio. The webhook gives only the caption and the public page URL. Downloading the video from that page would be scraping, which `AGENTS.md` §6 rules out. Options to weigh in SAV-7 and SAV-14:
   - index reels from the caption first, and treat the video as unavailable unless an official route exists;
   - check whether an official API returns a reel's media from `reel_video_id` for an app-scoped share;
   - get the video through capture paths that have the file (the Phase 3 share extension).

   **Result (SAV-28, record [0013](0013-dm-reel-media.md)):** in the tested setup (the bot's existing Unified-login token, development mode), the lookup by `reel_video_id` returned the video for a reel the bot owns. For the tested third-party reels and posts it returned an error whose cause wasn't established, and the message routes gave no media. Instagram Login with a valid token was not tested. DM-shared reels stay caption-only.
2. **Carousels: one image and no caption (decision needed).** Treat a carousel as a single image with no caption, unless an official way to get the other items and the caption turns up. Ask the user when that isn't enough to identify it.
3. **Post images must be downloaded right away.** The link works now, but there's no stated lifetime. Fetch and store the image (and a 360p copy, per §9) when the webhook arrives. Re-check the link after 24 hours to measure its life; still to do.
4. **The adapter (SAV-11):**
   - parse `entry[].messaging[]`;
   - accept `ig_reel`, `ig_post` and, to stay tolerant, the old `share`;
   - treat `title` as optional;
   - convert millisecond timestamps;
   - check the signature with the Meta App Secret;
   - attach a text sent within a short window after a share as its `collection_hint`; 9 s was observed, so the window should be configurable.
5. **Onboarding:** the token-and-resubscribe step from finding 10 goes into the README's setup steps when SAV-11 adds the real webhook. Finding 11 is re-tested with a second tester before launch.

## How to record again

1. Start a recorder that answers the `hub.challenge` handshake, checks `X-Hub-Signature-256`, and saves raw bodies **outside the repo**. Then start `ngrok http <port>`.
2. In the Meta app go to **Use cases → Customize → API setup with Unified login**:
   - in **Configure webhooks**, set the callback URL and verify token, then subscribe to `messages`;
   - in **Generate access tokens**, generate a token for the bot (don't copy it), then switch **Webhook Subscription** off and on.
3. DM the bot from an Instagram Tester account, then anonymize as described in `tests/fixtures/webhooks/README.md`. Delete the raw recordings afterwards.
