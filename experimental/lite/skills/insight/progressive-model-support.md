# Progressive Model Support

Add new model support in layers:

1. Register the model and a minimal protocol.
2. Build an importable config path.
3. Add a tiny forward path.
4. Add weight loading.
5. Add distributed execution and checkpointing.

Do not combine all layers in the first PR for a model family.
