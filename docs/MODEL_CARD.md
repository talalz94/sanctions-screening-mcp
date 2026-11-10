# Model card: sanctions name matcher

## What it is

A name matcher that screens a person's name against the people on sanctions lists and returns candidates with scores and a word by word explanation. It has three parts:

1. **Normaliser** (`normalize.py`). Arabic script (with Persian and Urdu letters) and Cyrillic (Russian, Ukrainian, Serbian, Macedonian and the Central Asian alphabets) are transliterated, and Latin letters with no accent to strip (ı, ł, đ, ø, ə) are spelled out. A name that mixes scripts is read one script at a time. Titles (Sheikh, Haji, Major General), articles (al, el, and the sun letter forms such as ad-Din and ar-Rahman), connectors (bin, ibn, ben, ould) and South Asian "s/o" are dropped; Hafiz is dropped only in front of two or more name words, because it is also a given name. "Abd al-X" compounds, "-ullah" and "-uddin" suffixes and about 50 groups of common given name spellings are unified. Each word gets a consonant skeleton key, plus a second key where a spelling can be read two ways: Latin th, sh, kh and dh (one Arabic letter or two), Latin dj (Djamel or Djordjevic), Latin gh (غ, or ق in Persian), Arabic ض (d in Arabic, z in Persian and Urdu), Arabic و as a consonant (w in Arabic, v in Persian), Cyrillic г (g in Russian, h in Ukrainian) and a leading al or ال (the article, or part of the name as in Alberto or Ilyas). Titles are dropped in Persian and Urdu spellings too (Haji, Maulvi, Mufti, doctor).
2. **Candidate search** (`index.py`). Each word is indexed under its exact keys and, for keys of four letters or more, under its one deletion keys. Exact key matches count fully, fuzzy ones 0.4, weighted by how rare the word is on the list. The best 200 people go on to scoring, not the best 200 names, so one person with many aliases cannot crowd out the rest. An initial retrieves nothing by itself and adds little weight.
3. **Scorer** (`scoring.py`). Words are aligned one to one regardless of order. Two words count as the same word when they are the same name after spelling variants are merged, or nearly so by Jaro Winkler similarity, or when they share a consonant key and the vowel pattern behind it (Kareem and Karim, not Karim and Akram). Two different common names that share consonants, such as Hasan and Husain, are never a strong match. Across scripts, where Arabic script leaves out short vowels, a pair counts as the same word only when it shares a key of two consonants or more and both sides agree on a final long i (Amir is not امیری, Amiri); an Arabic word recognised as a common name is compared as that name's Latin spelling, and a common name whose Arabic spelling is known never matches a different Arabic word in full. Two Arabic script words match in full only when they are spelled the same, because within Arabic script a name keeps its letters (سلمان Salman is not سليمان Sulaiman). Eleven features are computed from the alignment (one of them marks an Arabic script query against a Latin name), and either fixed weights or a logistic regression turns them into a score.

No neural model, no external service and no API calls. Everything runs on a laptop CPU.

## Training

The logistic regression is trained on queries of the train split only, on rows that mirror the screening decision, built from the same 200 candidates that screening scores:

- each train query that names a listed person gives one positive row: that person's best name under the fixed weights, when the person is among the candidates;
- each clean train query gives up to three negative rows: its top three candidate names under the fixed weights, the names most likely to raise false alerts;
- other candidates of a query that names a person are left out, because many are namesakes of that person and labelling them negative would add noise.

The alert threshold is then set on dev clean names. Weights are published in the README.

## Intended use

- Research and teaching: how name screening works and how to measure it.
- Decision support for a trained analyst who reviews every alert with the full record, birth date and nationality.

## Out of scope

- Any commercial screening without an OpenSanctions data license (see the data card).
- Automatic decisions. The matcher never clears or blocks anyone; it ranks candidates for a person to review.
- Organisations, vessels and aircraft (not indexed yet).

## Risks and limits

- **False alerts fall unevenly.** People with common names, many of them Arabic and South Asian names, collide with listed names far more often. The README reports false alerts by name origin so this stays visible.
- **Misses are the costly error in compliance.** Thresholds here are chosen for a false alert budget; a real deployment would set them from its own risk appetite and review capacity.
- **Transliteration rules are hand written.** They cover common Arabic, Persian, South Asian, Turkic and Slavic patterns, not every language. Names written only in other scripts are not indexed.
- The generated test variants share assumptions with the normaliser; see the data card.
