"""Check if a given slurm job id completed successfully
   Usage:
       python3 check_slurm_job_completion.py <JOB_ID>
"""

import sys
import subprocess


cmd = f"sacct -j {sys.argv[1]}"
result = subprocess.check_output(cmd, shell=True).decode().split()
assert len(result) > 14, "JOB state not available."

status = result[19]
exit_code = result[20]

assert status == "COMPLETED", f"Job {sys.argv[1]} not completed."
assert exit_code == "0:0", f"Job {sys.argv[1]} did not exit successfully."

