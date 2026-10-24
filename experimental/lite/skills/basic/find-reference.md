# Find Reference

Before porting a model or primitive, identify the source implementation and the
minimal behavior needed for the current PR.

Record:

- The upstream source file or paper section.
- The expected tensor shapes.
- The smallest local check that can catch a regression.

Do not copy large model stacks into Lite when a smaller contract check is enough.
