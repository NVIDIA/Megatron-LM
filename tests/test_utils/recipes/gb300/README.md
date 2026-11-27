# GB300 recipes

GitHub Actions uses the four-GPU `nvidia-ci-aws-use2-gpu-x4` runner as
`dgx_gb300`. The runner label and `gb-gpu` container image suffix are deployment
identifiers and remain unchanged.

`github-*.yaml` contains the GitHub functional suites, including disabled
entries, copied from the former GB200 lane. Existing recipes without that
prefix serve internal CI. `unit-tests.yaml` defines GitHub unit-test buckets.
GitHub selects the `launch_on_gb300` pytest marker. Internal GB200 unit launches
continue to select the same four-GPU suite.

GitHub goldens are byte-for-byte copies of the existing GitHub GB200-named
baselines, which were already recorded on GB300 hardware. The older files stay
available to internal GB200 recipes. Three cases already have distinct internal
GB300 baselines; their GitHub rows use `golden_platform: dgx_gb300_github` to keep
both recordings. Golden downloads preserve those output filenames.

The lane reads the repository variable `ENABLE_GB300_TESTING`, falling back to
`ENABLE_GB200_TESTING` when the new variable is unset. Setting the new variable
to the string `false` disables the lane even if the legacy variable is `true`.
Testmon uses `dgx_gb300` cache identities and the existing restoration and
fallback behavior; cache population publishes the new platform namespace.
