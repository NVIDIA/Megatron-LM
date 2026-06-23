#!/bin/bash

# SLURM by default terminates all user processes when the main job process is
# finished. This also immediately terminates inprocess.MonitorProcess and
# prevents it from submitting information to distributed store, and finalizing
# iteration by waiting on termination barrier.
#
# This script waits for all "python" processes launched by the current user to
# finish before terminating the SLURM job.

is_daemon_running() {
    pgrep -u $USER "python" > /dev/null
}

wait_daemon() {
   while is_daemon_running; do
      sleep 1
   done
}
