#!/usr/bin/env bash
# mac-health — read-only check of what heats the Mac (MAC-COOL, Lucas 2026-10-23). Step 0 of every
# session and the check before any heavy job. Exit 1 when something needs action. Never kills anything.
# Adapted from Friday's ~/PEBLE/tools/mac-health.sh; the mac-health plugin's /mac runs this file.
#
# It reports: load and macOS thermal pressure, the top CPU users, Spotlight, the Mac-wide heavy-job
# lock (shared with Friday), ICOR's dev servers, orphaned e2e servers, long ICOR jobs and worktrees
# that Spotlight would index.
set -uo pipefail

warn=0
bad() { echo "  ✗ $*"; warn=1; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NCPU=$(sysctl -n hw.ncpu)
L1=$(sysctl -n vm.loadavg | awk '{print $2}')
TP=$(notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print $2}')

echo "Load $L1 on $NCPU cores (keep under $(( NCPU * 8 / 10 )))"
awk -v l="$L1" -v n="$NCPU" 'BEGIN{exit !(l >= n*0.8)}' && bad "load is high — no new heavy job; see the top list"
echo "Thermal pressure ${TP:-?} (0 nominal, 1 moderate, 2 heavy, 3 trapping)"
[ -n "$TP" ] && [ "$TP" != 0 ] && bad "the Mac is under thermal pressure — start nothing heavy; tools/heavy.sh waits on its own"

echo "Top CPU:"
ps -Ao pcpu=,comm= -r | head -5 | awk '{c=$2; sub(/.*\//,"",c); printf "  %5.1f %%  %s\n", $1, c}'

mds=$(ps -Ao pcpu=,comm= | awk '/mds_stores|\/mds$/{s+=$1} END{printf "%d", s}')
echo "Spotlight: mds + mds_stores ${mds} % CPU"
[ "$mds" -gt 50 ] && bad "Spotlight is busy (${mds} %) — something indexable is churning"

if [ -d /tmp/epsa-heavy.lock ]; then
  echo "Heavy job running (Mac-wide lock): $(cat /tmp/epsa-heavy.lock/cmd 2>/dev/null)(pid $(cat /tmp/epsa-heavy.lock/pid 2>/dev/null))"
else
  echo "Heavy job running: none"
fi

echo "ICOR dev servers (stop the ones you started when done):"
lsof -nP -iTCP -sTCP:LISTEN -Fpn 2>/dev/null | awk '/^p/{p=substr($0,2)} /^n/{print p, substr($0,2)}' \
  | while read -r pid addr; do
      port=${addr##*:}
      case "$port" in 8000|5173|18[0-9][0-9][0-9]|19[0-9][0-9][0-9]) ;; *) continue ;; esac
      cwd=$(lsof -a -d cwd -p "$pid" -Fn 2>/dev/null | awk '/^n/{print substr($0,2)}')
      echo "  :$port  pid $pid  ${cwd/#$HOME/~}"
    done | sort -u

# An e2e server whose runner is gone has been reparented to launchd (pid 1).
orphans=$(ps -Ao pid=,ppid=,command= | awk '$2 == 1 && /scripts[.\/]e2e_app|run_e2e_dev/ {print "    pid " $1 "  " substr($0, index($0, $3), 100)}')
[ -n "$orphans" ] && { bad "orphaned e2e servers (kill them if no Playwright run is yours):"; echo "$orphans"; }

echo "ICOR processes running over 30 min (check each is still wanted):"
ps -Ao pid=,etime=,pcpu=,command= | awk -v root="$ROOT" '
  index($0, root) && $4 !~ /awk/ {
    n = split($2, t, /[-:]/); s = t[n] + 60 * t[n-1]
    if (n >= 3) s += 3600 * t[n-2]
    if (n == 4) s += 86400 * t[1]
    if (s > 1800) printf "    pid %-6s %10s  %5.1f %%  %s\n", $1, $2, $3, substr($0, index($0, $4), 110)
  }'

git -C "$ROOT" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print substr($0,10)}' | while read -r wt; do
  [ "$wt" = "$ROOT" ] && continue
  case "$wt" in */.*|*.noindex|*.noindex/*|/tmp/*|/private/tmp/*) continue ;; esac
  echo "  ✗ worktree $wt is indexed by Spotlight — move it under .claude/worktrees/ or a *.noindex folder"
done | grep . && warn=1

[ $warn = 0 ] && echo "OK" || echo "ACTION NEEDED (see ✗)"
exit $warn
