# Chrome Timeline Inspector

LLM training profile analysis tool for torch profiler Chrome timeline traces. Supports single-rank and multi-PP (Pipeline Parallel) analysis.

## Features

- **VPP Schedule Timeline**: Visualize forward/backward passes and P2P communication as Gantt charts
- **Module Performance**: TFLOPS and bandwidth analysis for attention, MoE, MLP, router, dispatch/combine modules
- **Multi-PP Timeline**: Multi-rank pipeline parallel visualization with bubble analysis and schedule table mapping
- **Event-level Details**: Drill down into individual module events with GPU kernel timing

## Quick Start

### Prerequisites

- Python with `perfetto`, `streamlit`, `plotly`, `pandas`, `numpy`, `pyyaml`
- `trace_processor_shell` binary [from Perfetto](https://github.com/google/perfetto/releases). [V50.1](https://github.com/google/perfetto/releases/tag/v50.1) has been tested




### Single-Rank Analysis (PP=1)

```bash
# Launch the Streamlit visualizer
streamlit run visualize_profile.py
```

In the sidebar:
1. Enter trace file path (`.pt.trace.json.gz`)
2. Enter `trace_processor_shell` binary path
3. (Optional) Enter model config path for TFLOPS calculation
4. Click "Load and Analyze"

### Programmatic Usage

```python
from process_profile import LLMProfileAnalyzer

analyzer = LLMProfileAnalyzer(
    trace_path='examples/pp1/trace_rank0.pt.trace.json.gz',
    bin_path='/path/to/trace_processor_shell',
    model_config_path='model_config.yaml',  # optional, for TFLOPS
)

# VPP schedule analysis (forward/backward timing)
vpp_result = analyzer.analyze_vpp_schedule()

# Module-level analysis (attention, MoE, MLP, etc.)
module_result = analyzer.analyze_modules()
```

### Multi-PP Analysis (PP > 1)

Place one trace file per PP rank in a directory. Files are sorted by `rank<N>` in the filename.

```
traces_pp4/
├── torch_profiler_rank0_*.pt.trace.json.gz
├── torch_profiler_rank1_*.pt.trace.json.gz
├── torch_profiler_rank2_*.pt.trace.json.gz
└── torch_profiler_rank3_*.pt.trace.json.gz
```

**Streamlit UI**: Enter the directory path in "Multi-PP Analysis" sidebar section, click "Load Multi-PP".

**Programmatic**:

```python
from process_profile import MultiPPAnalyzer

multi = MultiPPAnalyzer(
    trace_dir='traces_pp4/',
    bin_path='/path/to/trace_processor_shell',
    model_config_path='model_config.yaml',
)

result = multi.analyze_all()
# result['pp_degree'] = 4
# result['ranks'][0]['forward_steps'] = [{'start_ms': ..., 'mbs_id': 0, 'chunk_id': 0, 'label': 'F0c0'}, ...]
# result['summary']['bubble_ratio'] = 0.15
```

Key features:
- Parallel trace loading via ThreadPoolExecutor
- Time alignment: each rank's first forward GPU start aligned to global t=0
- Schedule table mapping: forward/backward steps labeled with `(mbs_id, chunk_id)` from VPP schedule table
- P2P direction extraction from NCCL kernel args (collective name, dst rank)
- Bubble analysis per rank

## File Structure

```
chrome_timeline_inspect/
├── README.md                   # This file
├── process_profile.py          # Core analysis engine
│   ├── LLMProfileAnalyzer      # Single-rank analyzer
│   ├── MultiPPAnalyzer         # Multi-PP analyzer
│   ├── ModelFlopsCalculator    # TFLOPS calculation from model config
│   ├── EventCorrelator         # CPU-GPU event correlation
│   └── SQLQueryBuilder         # Perfetto SQL query generation
├── visualize_profile.py        # Streamlit visualization app
├── module_flops.py             # FLOPs formulas (MLA, KDA, MoE, MLP, router, loss)
├── module_bandwidth.py         # Communication volume formulas (dispatch, combine)
├── model_config.yaml           # Example model config (YAML)
└── examples/
    ├── pp1/                    # Single-rank examples
    │   ├── *.pt.trace.json.gz  # Example trace file
    │   └── timeline_parse.ipynb # Usage notebook
    └── pp_multi/               # Multi-PP examples (add your traces here)
```

## Model Config

The `model_config.yaml` provides model architecture parameters for TFLOPS/bandwidth calculation:

- `model.config.hidden_size`, `num_attention_heads`, `seq_length`, etc.
- `trainer.args.extra_configs.micro_batch_size`, `swiglu`, etc.
- `trainer.args.extra_configs.pipeline_model_parallel_layout` (for VPP chunk detection in multi-PP)

For multi-PP, the `pipeline_model_parallel_layout` string (e.g., `Et*4|t*5|t*5|t*2mL`) determines `num_model_chunks` (VPP size).

## Analyzed Modules

| Module | Metric | Description |
|--------|--------|-------------|
| KimiDeltaAttention / MLASelfAttention | TFLOPS | Attention layer (KDA, MLA, or standard) |
| MoELayer | TFLOPS | Full MoE layer (experts + shared + router) |
| TEGroupedMLP | TFLOPS | Routed experts only (GroupGEMM) |
| SharedExpertMLP | TFLOPS | Shared expert MLP |
| TopKRouter | TFLOPS | Router linear projection |
| FusedDispatch | Bandwidth (GB/s) | Token dispatch communication |
| FusedCombine | Bandwidth (GB/s) | Token combine communication |
| MLP | TFLOPS | Dense MLP |
| MultiTokenPredictionBlock | TFLOPS | MTP = attention + MoE + logits |
