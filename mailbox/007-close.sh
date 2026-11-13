#!/bin/bash
# Job 007: job 006 didn't stop the watcher, so stop it by process id.
echo "before:"; ps -axo pid,ppid,command | grep -i "mailbox.sh" | grep -v grep
for pid in $(pgrep -f "agent/lab/mailbox.sh"); do echo "TERM $pid"; kill -TERM "$pid"; done
sleep 3
for pid in $(pgrep -f "agent/lab/mailbox.sh"); do echo "KILL $pid"; kill -KILL "$pid"; done
sleep 1
echo "after:"; ps -axo pid,ppid,command | grep -i "mailbox.sh" | grep -v grep || echo "mailbox closed"
