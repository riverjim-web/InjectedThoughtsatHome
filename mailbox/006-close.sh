#!/bin/bash
# Job 006: all experiments are finished, so close the mailbox early instead of leaving it
# open until 09:28. This stops the watcher (and the caffeinate wrapped around it).
echo "$(date '+%F %T') closing the mailbox: jobs 001-005 are done; nothing else will run."
pkill -f "zsh $HOME/agent/lab/mailbox.sh" || pkill -f "mailbox.sh"
