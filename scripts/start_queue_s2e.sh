#!/bin/bash
# waits until s2d has started and finished (avoids the race between s2c end and s2d start), then runs the reasoning-probe 2x2 cells
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2d.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2d[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E6d_varchain_loop3fixed_moe_engram.json $C/E8c_add_moe_engram.json $C/E8d_add_loop3fixed_moe_engram.json $C/E8e_add_moe_core6.json $C/E8f_add_moe_core6_blockcos.json $C/E4t_hyb8_blockcos.json > experiments/queue_s2e.log 2>&1 < /dev/null &
