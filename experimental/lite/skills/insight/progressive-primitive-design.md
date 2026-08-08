# Progressive Primitive Design

A primitive should move from contract to reference to optimized implementation.

The contract defines behavior. The reference proves correctness. The optimized
implementation proves performance without changing the contract.

When these steps are split, reviewers can evaluate correctness before performance
complexity enters the diff.
