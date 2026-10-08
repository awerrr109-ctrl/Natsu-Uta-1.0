#!/bin/bash
# F006 mechanism: run a foreground command only if it cannot OOM-kill a queue job.
# usage: scripts/fg_guard.sh <need_mb> <command...>
need=$1; shift
if pgrep -f "natsu[.]train" >/dev/null; then
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  peak=$(ps -o rss= -p "$(pgrep -f 'natsu[.]train' | head -1)" | awk '{print int($1/1024)}')
  # the queue job may still grow toward its config's known peak (~850 MB worst case observed): reserve 450 MB headroom
  if [ $((avail - 450)) -lt "$need" ]; then
    echo "FG_GUARD: refused (avail ${avail} MB, job rss ${peak} MB, need ${need} MB + 450 headroom). Queue it instead." >&2
    exit 75
  fi
fi
exec "$@"
