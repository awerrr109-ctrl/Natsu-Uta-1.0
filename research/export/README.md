Compressed export of `research/db/research.sqlite` (abstracts omitted to keep the repo small; they are re-fetchable).
The live DB is git-ignored (72 MB). Rebuild: `python research/import_export.py` (restores tables from these files), then
`python research/harvest.py all` only fetches queries not yet in `queries_done`.
