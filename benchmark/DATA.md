# Where the insults come from

> [!WARNING]
> `insults.jsonl` contains slurs, sexual content and threats, uncensored.

`insults.jsonl` has 450 insults:

- 150 `twitter` rows are tweets labelled hate speech or offensive in [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) under MIT license
- 150 `comments` rows are news-site comments with an insult score of at least 0.85 in [google/civil_comments](https://huggingface.co/datasets/google/civil_comments), which is CC0
- 150 `gaming` rows are CS2 chat lines from the same scrape EBLAN was trained on, held out so the model never saw them

Check the terms on those pages yourself before relying on this summary.

Each row has an `id`, a `category`, the `insult`, and two ratings from the curation step, `vicious` and `effort`.

## How they were picked

They were filtered through a script that keeps lines that talk to the reader ("you", "your", "u" and so on), drops links, mentions and duplicates, and keeps them short (5-45 words for tweets, 5-50 for comments and 5-40 for chat). Chat lines with game words in them were dropped too, so they read like general trash talk.

Then an LLM rated every candidate. It decided whether the line is really an attack on the person it's talking to (so a reply makes sense without more context), how vicious it is from 0 to 10, and how much effort went into it from 0 to 10. Only the real attacks were kept, they were ranked by `2 * vicious + effort`, and the top 150 of each source made it in.

## What was changed

The text is left as it was, slurs included. The only edits are for privacy, which is why you may see `[username]`, `[name]` and `[address]`. There's no public figures. A few rows have `�` characters that were already in the source.

## License

The texts themselves stay under the terms of their sources above. The repository's MIT license covers the code, not these texts.

## Removal requests

If one of the `gaming` lines is yours and you want it gone, open an issue or email contact@outpoot.com and it'll be replaced.
