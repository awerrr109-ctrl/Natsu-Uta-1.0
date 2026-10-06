# F006 (ops) — concurrent queue runners + torchrun test → CPU thrash and OOM kill

- **what happened**: restarting `start_queue_s2a.sh` several times left an older runner alive. Two experiments trained concurrently
  (E5a and E5b, ~270 MB RSS each, load avg 12). A 2-process torchrun test added more load and was OOM-killed (exitcode −9). B3 and E4k barely ran.
- **impact**: no wrong numbers were recorded (the partial runs had no final.json and were deleted and re-queued). About 25 min of compute was lost.
- **root causes**: (1) waiter scripts restarted without killing the runner they had already spawned; (2) `pgrep -f "natsu.train"`
  matched the tool's own shell command line (F-ops note in state); (3) no global lock.
- **fix**: `scripts/start_queue_s2c.sh` refuses to start if any `run_queue.sh` is alive (regex `run_queue[.]sh` avoids the self-match);
  heavy one-off tests run only when the queue is idle.
- **lesson**: on a 1 GB / 2 vCPU box, concurrency is an experimental confound (throughput, tok/s, and RSS numbers) as well as a crash risk.

## Recurrence (session 2, 01:50 UTC) — caused by the agent
- A foreground 3-step smoke test (E8f, 1.6M params, peak ~480 MB) ran while the queue's E4o (peak ~720 MB) was training on a 985 MB box.
  The global OOM killer took E4o at step ~150. Same mechanism as before: "small" ad-hoc jobs are not small relative to 1 GB.
- **Rule (now binding)**: no foreground model instantiation while a queue job runs, unless `free -m` available > peak_rss(queue job) + 500 MB.
  Smoke tests go into the queue as their own entry. E4o requeued at the front of s2e.

## Second recurrence (session 2, 07:27 UTC) — agent again; the written rule was not enough
- A foreground `import torch` plus a meta-device model build (for a param count) while E4l (peak 842 MB) trained → OOM killed E4l at step 700/800.
- **Why the rule failed**: it relied on me checking `free -m` by hand before every command. A meta-device build "felt" free, but torch import alone is ~200 MB RSS.
- **Fix (mechanism, not intention)**: `scripts/fg_guard.sh <need_mb> cmd…` refuses (exit 75) when a training job runs and MemAvailable − 450 MB < need.
  All foreground python that imports torch now goes through it. E4l is requeued (s2h).
