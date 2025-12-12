# Welcome to Megatron-LM

**Megatron-LM** is a powerful, open-source deep learning library developed by **NVIDIA** designed to train massive foundational models that are too large to fit into the memory of a single GPU.

Built on top of PyTorch, it solves the challenge of **scaling multi-billion (and even trillion) parameter models** by employing efficient model parallelism techniques. This approach significantly reduces memory constraints and communication overhead, enabling researchers to train state-of-the-art transformer models—such as GPT, BERT, and T5 efficiently across **thousands of GPUs**.

## Core Capabilities

- **Model Parallelism:** Overcomes single-GPU memory limits by leveraging several basic and advanced parallelism techniques.
- **Scalability:** Optimized to scale efficiently from a single GPU to thousands of GPUs.
- **PyTorch Integration:** Built seamlessly on top of PyTorch, supporting standard architectures like GPT, BERT, and T5.

## Performance Results

Brief section providing evidence Megatron is THE best-in-class training framework.

```{toctree}
:caption: Getting Started
:hidden:

getting-started/fifteen_minutes
getting-started/first_training_run
getting-started/faq
```


```{toctree}
:caption: User Guide
:hidden:

user-guide/basic/index
user-guide/intermediate/index
user-guide/advanced/index
```

```{toctree}
:caption: Developer Guide
:hidden:

developer-guide/submit_pr
developer-guide/create_issue
developer-guide/review_pr
```

```{toctree}
:caption: API Reference
:hidden:

apidocs/index
```

```{toctree}
:caption: Community
:hidden:

community/roadmap
community/office_hours
```

