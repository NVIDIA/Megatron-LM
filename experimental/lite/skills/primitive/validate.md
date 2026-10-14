# Primitive Validation

Validation should scale with the primitive risk:

- Interface-only changes need import and registry tests.
- Pure PyTorch primitives need numerical parity tests.
- Distributed primitives need CPU-safe unit coverage plus real Slurm evidence
  for GPU paths.

Keep the validation command stable so follow-up PRs can add tests without
changing reviewer workflow.
