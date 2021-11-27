
# Multi-Stage Prompting for Knowledgeable Dialogue Generation

We present the steps to run our multi-stage dialogue prompting (MSDP), as well as the baselines, finetuning-based knowledge generation (FKG) and finetuning-based coversation model (FCM).

## Multi-Stage Dialogue Prompting (MSDP)

### Data Preparation
1. Dataset Download: [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) and [Wizard of Internet](https://parl.ai/projects/sea/)
2. Data Processing: We provide script ```tasks/knwl_dialo/scripts/data_processing.sh``` to process the data.

### Knowledge Generation
1. The script ```tasks/knwl_dialo/scripts/prompt_knwl_gen.sh``` provides an example for how to perform the knowledge generation prompting.
2. The F1 score can be evaluated through ```tasks/knwl_dialo/scripts/eval_generation.sh```. Other automatic metrics follow the [nlg-eval](https://github.com/Maluuba/nlg-eval).

### Response Generation
1. Prepare the input file for the response generation (based on the previously generated knowledge file):
2. The script ```tasks/knwl_dialo/scripts/prompt_resp_gen.sh``` provides an example for how to perform the response generation prompting.
3. The automatic evaluations are the same as mentioned aboved for the knowledge generation.

## FKG

### Knowledge Generation

### Response Generation

## FCM

### Knowledge Generation

### Response Generation

