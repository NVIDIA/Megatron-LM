"""
Arguments:
- `experiment_directory` (containing a list of iter_{} subdirectories)
- `repo_name`: name of the repo to push to.
Convert all intermediate checkpoints from a `experiment_directory` and push to a single repo `repo_name` on the hub.
Each intermediate checkpoint is associated a different branch (for example) corresponding to its iteration.
This script could use `{experiment_directory}/hf_checkpoints/iter_{}` as local path to save the converted checkpoints, and push those for example.
"""


