
# Multi-Stage Prompting for Knowledgeable Dialogue Generation

Blow we present the steps to run our multi-stage dialogue prompting (MSDP) framework.

## Multi-Stage Dialogue Prompting

### Data Preparation
1. Dataset Download: [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) and [Wizard of Internet](https://parl.ai/projects/sea/)
2. Data Processing: We provide the script [`tasks/knwl_dialo/scripts/data_processing.sh`](./scripts/data_processing.sh) to process the data.

### Stage-1: Prompting for Knowledge Generation
1. The script [`tasks/knwl_dialo/scripts/prompt_knwl_gen.sh`](./scripts/prompt_knwl_gen.sh) provides an example for how to perform the first-stage prompting for the knowledge generation.
2. We provide the script [`tasks/knwl_dialo/scripts/eval_knwl_generation.sh`](./scripts/eval_knwl_generation.sh) for the automatic evaluation (i.e., F1, BLEU, METEOR, and ROUGE-L) of the knowledge generation.

### Stage-2: Prompting for Response Generation
1. The script [`tasks/knwl_dialo/scripts/prep_resp_gen.sh`](./scripts/prep_resp_gen.sh) helps to prepare the input file for the response generation (based on the previously generated knowledge file).
2. The script [`tasks/knwl_dialo/scripts/prompt_resp_gen.sh`](./scripts/prompt_resp_gen.sh) provides an example for how to perform the second-stage prompting for the response generation.
3. We provide the script [`tasks/knwl_dialo/scripts/eval_resp_generation.sh`](./scripts/eval_resp_generation.sh) for the automatic evaluation (i.e., F1, KF1, BLEU, METEOR, and ROUGE-L) of the knowledge generation.
