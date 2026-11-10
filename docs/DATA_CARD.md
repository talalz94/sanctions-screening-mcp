# Data card: OpenSanctions snapshot and the screening eval set

## Source

| | |
|---|---|
| Publisher | [OpenSanctions](https://www.opensanctions.org) |
| Collection | `sanctions`, one immutable snapshot: `20261109194706-bbu` |
| File | `targets.simple.csv` (79 MB), SHA-256 pinned in `src/sanctions_mcp/data.py`, SHA-1 as published in the snapshot index |
| License | Creative Commons Attribution NonCommercial 4.0 (CC BY-NC 4.0) |
| How this repo uses it | Downloaded by `sanctions download` after you accept a notice. Never committed. Only aggregate metrics are published here. |
| Personal data | Yes. Names, birth dates and countries of people on public sanctions lists. |

## License and use

OpenSanctions data is free for noncommercial use. OpenSanctions says that any use inside a business needs a data license, and that compliance screening counts as commercial use even when it earns no revenue. This repo is a personal, noncommercial research project: it is not run for, or inside, any business. If you want to screen real customers with it, buy a data license from OpenSanctions first.

Attribution: data from OpenSanctions, licensed CC BY-NC 4.0.

The download is pinned to this one snapshot and its SHA-256, so every number here can be reproduced exactly. OpenSanctions keeps dated snapshots downloadable for a limited time. Once this one is gone, `sanctions download` stops with a message that says so, and the way forward is to set `SNAPSHOT` and `SHA256` in `src/sanctions_mcp/data.py` to a current snapshot and run everything again. The numbers will then move a little.

## What is used

People only (`schema = Person`). Organisations, vessels and other entity types are left out of this slice.

Every name and alias of every person is read. Words in Latin, Arabic (with Persian and Urdu letters) and Cyrillic script become tokens; Latin letters with no accent to strip, such as ı, ł, đ, ø and ə, are spelled out first. A name with no word in those scripts (for example one written only in Chinese, Greek, Georgian or Hebrew script) gives no tokens and is not indexed, so that person can only be found through their other names. Names of one person that are identical once normalised (for example "Mohammed ALI" and "ALI, Muhammad") are indexed once. `uv run sanctions stats` prints the counts.

## The eval set

Built by `uv run sanctions build-evalset` from the snapshot and seed 13, deterministically. It is never committed, because it is derived from CC BY-NC data.

- **Splits.** People are split 60/20/20 into train, dev and test by a hash of their id, and each split's queries are built from that split's people only. The scorer is trained on train queries, thresholds are set on dev, and only test numbers are reported in the README. The index, the word rarity weights and the candidates searched during training cover the whole list, as they would in production.
- **Real aliases.** For a sampled person with at least two Latin script names, one alias is held out of the index and used as the query. It must be a spelling variant of a name that stays in the index (difflib ratio of at least 0.6 on a plain normalisation), so the query is a variant the list publishers really recorded, not a nickname that no name matcher could link. An alias is never used when another name of the same person differs from it only in case, accents, punctuation or word order, because that twin would stay in the index and give the answer away.
- **Arabic script.** For a sampled person with both Arabic script and Latin names, one Arabic script name is the query and every Arabic script name of that person is held out, so a match has to cross scripts.
- **Generated variants.** Up to two per sampled person, from `src/sanctions_mcp/variants.py`: transliteration changes, typos, a dropped middle name, an initial, an article or connector added or removed, a title added, compound spacing. A variant that equals one of the person's kept names after plain normalisation is discarded.
- **Clean names.** One or two given names of one listed person (titles and particles skipped) joined to the family name of another listed person from the same name origin group, never the same person and never a person with the same family name. Every alert on a clean name is a false alert, so a clean name is discarded when a name only screener could not tell it apart from someone real:
  - it is a listed name, a query that should alert (in any split), or an earlier clean name, under the matcher's own idea of the same word: the same common name after spelling variants are merged (Mohammed and Muhammad), or the same consonant key with the same vowel pattern (Fadel and Fadil), or, against Arabic script, the same consonant key;
  - it is a listed name with middle names left out (the first one or two words and the last word of a name of three words or more);
  - it has fewer than two words once titles and particles are dropped.

  Two different common names that merely share consonants, such as Hamid Ali and a listed Ahmad Ali, are kept as clean names, so the false alert rate counts them.

  Most clean names are in Latin script. A further 1,000 per split are built the same way in Arabic script, from the Arabic script names of listed people (titles, connectors and the family article آل skipped), so a scorer pays for false alerts on Arabic script queries as well. They are checked against every listed name as an Arabic script query would see it, Latin script names included, so none is a namesake of someone on the list. Without them, nothing in the eval would stop a learned scorer from lifting every Arabic script query.
- **Name origin groups.** "mena_south_asia" when any of the person's listed countries is one of the 22 Arab League members, Iran, Afghanistan or Pakistan, or when they have an Arabic script name; otherwise "other". People listed only under countries such as India, Bangladesh, Turkey or Israel are "other" unless one of their names is in Arabic script. The group pairs given and family names for clean names and slices the results. The matcher never sees it.

## Known limits

- The generated variants come from rules that share assumptions with the normaliser (for example that vowels drift between transliterations). They are an easier test for this matcher than for the baselines. The real alias and Arabic script rows are the fairer comparison.
- Clean names are made from listed names, so they look like the names a screener sees on alert, not like a random customer base. False alert rates on a real customer base would differ.
- Birth dates and nationalities are not used for scoring. In practice they clear many name only alerts.
- The list changes several times a day. A pinned snapshot keeps the numbers reproducible but goes stale.
