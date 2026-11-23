# 0017: Filters from extracted values when no entity is linked

- Status: accepted. Muhammed approved the ARCHITECTURE change by creating SAV-35 (23 Nov 2026).
- Date: 2026-11-23
- Ticket: SAV-35 (builds on SAV-16's record 0014; record 0005 sets the global, any-language rules)

## Context

On 23 Nov 2026 Muhammed moved the Google Places and TMDB resolvers (SAV-15) and the "which one?" question to Phase 2. Many places share a name, so a match must be tested before Savely relies on it. Phase 1 promises something simpler: "show me the reels I saved in Kadıköy" returns those reels, without Maps data.

Search's place filters only read data a resolver creates. `area` read the linked entity's address levels, and `cuisine` the entity's data or that entity's own mention. Without SAV-15 no place has an entity, so both found nothing and search fell back to keywords (L1/L2) with an "I couldn't filter exactly" notice.

The extraction already stores what the reel said for each mention, in `saved_item.fields.entities[i]`: a place's `area_hint` ("Kadıköy", "Brooklyn, New York", "渋谷、東京") and its vocabulary-normalized `cuisine`.

## Decision

1. **Filters are checked per mention.** All of a search's type filters must hold on the same mention, as before. A mention **with** a linked entity (`item_entity.position = i`) is judged exactly as before, by the entity's verified data. A mention **without** one is judged by the mention itself (`unresolved_mention()` in `app/agent/search/conditions.py`, an `EXISTS` over `jsonb_array_elements(fields.entities) WITH ORDINALITY` that skips positions with an `item_entity` row):
   - a value, integer, boolean or text filter (`cuisine`) reads the mention's key named after the filter, the same key `_own_mention()` reads for a resolved mention;
   - an area filter reads the parts of the mention's hint (below);
   - filters that need an entity (`near`, `open_now`, `streaming`, `watched`) never hold for it.
2. **The content type says which mention key holds the area.** `SearchFilter` gets `mention_field`, allowed only on `AREA` filters, and `ContentType` checks that the type's mentions have that field. `place` declares `area`'s `mention_field="area_hint"`. Search names no type and no key.
3. **A hint's levels are its parts, split on commas and semicolons of every script.** The separators are the Unicode "other punctuation" marks whose name contains `COMMA` or `SEMICOLON` (36 marks, e.g. `,` `;` `،` `؛` `、` `，` `；` `､`), read from Python's Unicode database rather than listed by hand, and passed to Postgres as a regular expression (`area_hint_separator_pattern()`). White space around a separator is trimmed. "Kadıköy, İstanbul" → "Kadıköy", "İstanbul"; "渋谷、東京" → "渋谷", "東京"; "الزمالك، القاهرة" → "الزمالك", "القاهرة". An area matches when it equals a whole part, after normalization.
   - Hyphens, slashes, apostrophes and full stops are **not** separators, because they appear inside names ("Saint-Germain-des-Prés", "St. John's"). A hint written without separators ("東京都渋谷区", "Moda - Kadıköy") stays one part and only matches as a whole.
   - The whole hint is never a value of its own: "Kadıköy, İstanbul" is matched through its parts.
4. **Normalization offers the same parts.** The area values of the asking user (`_user_areas` in `saved_values.py`) are their linked entities' address levels plus the parts of the hints of their mentions with no entity, so "kadıköy" is an exact hit for a saved "Kadıköy", and "kadikoy" is offered "Kadıköy" as an option. A resolved mention's hint is not offered, because it no longer decides anything. Only the user's own items are read. SAV-16's rules are unchanged: exact hits skip the model, `FILTER_NORMALIZATION_MAX_OPTIONS`, `FILTER_NORMALIZATION_MIN_PROBABILITY` and the prompt stay as they are. Cuisine options already came from the list at the top of `fields`, which includes unresolved mentions.
5. **The entity wins once it exists.** When SAV-15 links an entity to mention `i`, that mention is judged by the entity's address and data, and its hint is no longer read or offered. SAV-15 has nothing to undo.
6. **L2 widens only what an entity knows.** A district or neighbourhood of a linked place is still widened to its city. An area known only from hints stays as it is at L2, because a hint doesn't say reliably which level each part is.

## Alternatives considered

- **Store the parsed parts** (a column or a key in `fields`) at indexing time: faster to filter, but a schema or pipeline change, more stored data for "delete my data", and a re-index of existing items. Splitting at query time over one user's items is fast enough (the SAV-16 speed test still passes).
- **Ask the text model for structured levels** (`neighbourhood`, `city`, …) instead of one hint: changes the extraction prompt and needs the SAV-19/20 evals. The model also often can't tell a district from a city.
- **Match an area anywhere in the hint text** (`ILIKE '%Moda%'`): would match "Moda" inside "Modan Street" or "Mod" inside "Moda". Equality with whole parts keeps the filter exact, and normalization handles spelling.
- **Split on every punctuation mark** (`[[:punct:]]`): breaks names with hyphens, apostrophes and full stops.
- **A hand-written list of separators**: would miss scripts nobody thought of; the Unicode names cover every script's commas and semicolons.
- **Movie genre from the extraction:** `MovieMention` has no genre field, so "horror movies" still matches only through tags, category and text. Adding one changes the extraction prompt and is out of scope (SAV-35).

## Consequences

- "Show me the reels I saved in Kadıköy" is answered at L0 from the reel's own words, with no fallback notice, in any language and script.
- Hints are only as good as the extraction. A reel that names the wrong area is found under that area until a resolver links the place (Phase 2).
- Nothing new is stored, and no schema, dependency, model call or cost is added. "Delete my data" is unaffected: hints live in `saved_item.fields`, which it already deletes.
