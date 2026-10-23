#!/usr/bin/env bash
# heavy — run a heavy job ONE AT A TIME on the whole Mac, when the Mac is cool, under nice
# (MAC-COOL, Lucas 2026-10-23). Adapted from Friday's ~/PEBLE/tools/heavy.sh (MAC-COOL-1).
#
# WHY: the Mac has 15 fast cores and no efficiency cores, so `nice` alone does not cool it; only
# doing less at once does. Every heavy job, from ICOR or from Friday, queues behind ONE Mac-wide
# lock (/tmp/epsa-heavy.lock: the same lock and the same pid/cmd files as Friday, which the
# mac-health status line shows). It starts only when macOS reports no thermal pressure and the
# load is below 80 % of the cores. The PreToolUse hook tools/hooks/mac_load_guard.py refuses a
# heavy command that does not go through this script.
#
# Heavy = full pytest, full Vitest, the web build, any Playwright run, and the long data jobs
# (scripts/build_*, rebuild_*, benchmark_*, backtest_*, calibrate_*, measure_*, verify_client_release
# and the like). Targeted tests do NOT need it.
#
# USAGE (from an agent session, run it in the background, because the wait can be long):
#   tools/heavy.sh uv run pytest -q
#   tools/heavy.sh npm --prefix web run e2e
set -uo pipefail

[ $# -gt 0 ] || { sed -n '2,18p' "$0"; exit 2; }

# HEAVY_TEST_* are for tests/tools/test_heavy_sh.py only: a private lock, and no wait on the
# real load or heat.
LOCK=${HEAVY_TEST_LOCK:-/tmp/epsa-heavy.lock}
LOAD_WAIT_MAX=${HEAVY_TEST_LOAD_WAIT_MAX:-600}  # the load alone never holds a job more than 10 min
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN)
MAXLOAD=$(( NCPU * 8 / 10 ))
load1() {
  { sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}'; } || awk '{print $1}' /proc/loadavg
}
load1_int() { load1 | awk '{printf "%d", $1}'; }
# macOS thermal pressure: 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping. No sudo needed.
thermal() {
  if [ -n "${HEAVY_TEST_THERMAL:-}" ]; then echo "$HEAVY_TEST_THERMAL"; return; fi
  command -v notifyutil >/dev/null 2>&1 || { echo 0; return; }
  notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print ($2 == "" ? 0 : $2)}'
}

waited=0
until mkdir "$LOCK" 2>/dev/null; do
  holder=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
  [ $((waited % 60)) = 0 ] && echo "heavy: waiting ${waited}s — busy with: $(cat "$LOCK/cmd" 2>/dev/null) (pid ${holder:-?})" >&2
  sleep 5; waited=$((waited + 5))
done
echo $$ > "$LOCK/pid"; printf '%s ' "$@" > "$LOCK/cmd"
trap 'rm -rf "$LOCK"' EXIT
trap 'exit 130' INT TERM

# Real heat: no time limit. A job is never added to a Mac that macOS already reports as hot.
tw=0
while [ "$(thermal)" != 0 ]; do
  [ $((tw % 60)) = 0 ] && echo "heavy: thermal pressure $(thermal) (0 = nominal) — waiting ${tw}s for the Mac to cool" >&2
  sleep 10; tw=$((tw + 10))
done

lw=0
while [ "$(load1_int)" -ge "$MAXLOAD" ] && [ $lw -lt "$LOAD_WAIT_MAX" ]; do
  [ $((lw % 60)) = 0 ] && echo "heavy: load $(load1) ≥ $MAXLOAD on $NCPU cores — waiting" >&2
  sleep 10; lw=$((lw + 10))
done
[ "$(load1_int)" -ge "$MAXLOAD" ] && echo "heavy: load still $(load1) after ${lw}s; starting anyway (only heavy job, under nice)" >&2

start=$(date +%s); before=$(load1)
nice -n 10 "$@"
rc=$?
echo "heavy: done rc=$rc in $(( $(date +%s) - start ))s · load $before → $(load1) ($NCPU cores)" >&2
exit $rc
