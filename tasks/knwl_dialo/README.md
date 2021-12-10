
# Multi-Stage Prompting for Knowledgeable Dialogue Generation

Blow we present the steps to run our multi-stage dialogue prompting (MSDP) framework.

## Multi-Stage Dialogue Prompting

### Data Preparation
1. Dataset Download: [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) and [Wizard of Internet](https://parl.ai/projects/sea/)
2. Data Processing: We provide the script to run the [`data processing`](../../examples/knwl_dialo/data_processing.sh).

### Stage-1: Prompting for Knowledge Generation
1. We provide the script to perform the [`first-stage prompting`](../../examples/knwl_dialo/prompt_knwl_gen.sh) for the knowledge generation.
2. We provide the [`evaluation script`](../../examples/knwl_dialo/eval_knwl_generation.sh) for the automatic evaluation (i.e., F1, BLEU, METEOR, and ROUGE-L) of the knowledge generation.

### Stage-2: Prompting for Response Generation
1. We provide the script to [`prepare the input file`](../../examples/knwl_dialo/prep_resp_gen.sh) for the response generation (based on the previously generated knowledge file).
2. We provide the script to perform the [`second-stage prompting`](../../examples/knwl_dialo/prompt_resp_gen.sh) for the response generation.
3.  We provide the [`evaluation script`](../../examples/knwl_dialo/eval_resp_generation.sh) for the automatic evaluation (i.e., F1, KF1, BLEU, METEOR, and ROUGE-L) of the response generation.
