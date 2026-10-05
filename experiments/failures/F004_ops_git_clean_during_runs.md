# F004 (ops) — `git clean` during running experiments deleted live output files

- **what happened**: while creating an orphan `main` branch to give the PR a base, `git rm --cached . && git clean -fd` was run in the
  working tree that the experiment queue writes to. Committed files were restored by checking the branch out again, but the
  **open output files of the running job (E3b) were unlinked**, and the paper harvester process exited.
- **recovery**: outputs were copied back from `/proc/<pid>/fd/{1,3}` by a watcher until the job finished
  (`*.recovered`). SQLite DB integrity check: ok (32,381 papers, 14,205 repos, 20 reads). The harvester was restarted. Its queries are idempotent (`queries_done`).
- **lesson**: never run destructive git operations in the live working tree. Use `git worktree add /tmp/x` or a separate clone
  for branch surgery. Results directories should be git-ignored or written outside the repo by long-running jobs.
- **action**: none of the result numbers were affected. E3b's final.json is written at process exit to a fresh (non-deleted) directory path.
