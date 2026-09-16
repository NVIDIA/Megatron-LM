# Porting Guidance

Use this package as a staging area for small Megatron-native model ports.

Recommended order:

1. Register the model and a single implementation in `model.registry`.
2. Define a protocol module that returns a `ModelBundle`.
3. Add only the primitive contracts needed by that model.
4. Add CPU import and toy-level contract checks before GPU validation.
5. Move distributed, checkpoint, and downstream framework integration into
   separate PRs.

Avoid mixing model bring-up with unrelated runtime features. The main review
question for a model PR should be whether the model implementation follows the
contract and whether its validation evidence covers the behavior it adds.
