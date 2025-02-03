### Megatron Core Inference Documentation
This guide provides an example for Megatron Core for running model inference. 

### Contents
- [Megatron Core Inference Documentation](#megatron-core-inference-documentation)
- [Contents](#contents)
  - [1. Quick Start](#1-quick-start)
    - [1.1 Understanding The Code](#11-understanding-the-code)
    - [1.2 Running The Code](#12-running-the-code)
  - [2. Flow of Control In MCore Backend](#2-flow-of-control-in-mcore-backend)
  - [3. Customizing The Inference Pipeline](#3-customizing-the-inference-pipeline)
    - [3.1. Create Your Own Inference Backend](#31-create-your-own-inference-backend)
    - [3.2. Create Your Own Text Generation Controller](#32-create-your-own-text-generation-controller)
    - [3.3. Support Other Models](#33-support-other-models)
    - [3.3. Modify Inference Parameters](#33-modify-inference-parameters)
  - [4. Future work](#4-future-work)

<br>

#### 1. Quick Start
This example runs batch inference on a GPT model trained using Megatron Core. The entrypoint is [simple_gpt_batch_inference.py](./gpt/gpt_batch_inference.py)

<br>

##### 1.1 Code Walkthrough 
***STEP 1 - Initialize model parallel and other default arguments***
The micro batch size is set as 1 as it is not used in tensor-parallelism only, and for pipeline-parallel models it is calculated at runtime. 
```python
    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True, 'micro_batch_size': 1}
    )
```

***STEP 2 - Load the model using the model_provider_function***
NOTE: The model provider function supports both MCore and Legacy models. 

```python
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]
```

***STEP 3 - Choose an engine***
Text generation requires an inference engine, which includes a scheduler. The default engine is the [Megatron Core engine](../../megatron/core/inference/engine/mcore_engine.py) with a simple [text generation controller](../../megatron/core/inference/text_generation_controllers/text_generation_controller.py). TRTLLMEngine will be supported in the future.
```python
    inference_wrapped_model = GPTInferenceWrapper(model, args)
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, 
        tokenizer=tokenizer
    )
    inference_backend = MCoreEngine(
        text_generation_controller=text_generation_controller, max_batch_size=args.max_batch_size
    )
```

***STEP 4 - Run text generation***
The [SamplingParams](../../megatron/core/inference/sampling_params.py) contains suggested defaults. Customize this to change top_p, top_k, number of tokens to generate etc. 
*Note: The result is returned as a list of [InferenceRequests](../../megatron/core/inference/inference_request.py)*
```python
    results: List[InferenceRequest] = inference_engine.generate(
        prompts=args.prompts, sampling_params=sampling_params
    )
    
    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' ------------- RESULT FOR PROMPT {idx} --------------- ')
            result = {
                'id': result.request_id,
                'input_prompt': result.prompt, 
                'generated_text': result.generated_text,
                'generated_tokens' : result.generated_tokens
                }
            print(result)
```

<br>

##### 1.2 Running The Code
An example run script is shown below. Set the tokenizer paths, inference params, and other settings appropriately. 

For a quick recap on sampling parameters, refer to [this blog](https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910).

```
# In a slurm cluster (You could also use docker)
ACCOUNT=<account>
MLM_PATH=/path/to/megatron-lm
GPT_CKPT=/path/to/gpt/ckpt
VOCAB_MERGE_FILE_PATH=/path/to/vocab/and/merge/file
CONTAINER_IMAGE=nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11

srun --account $ACCOUNT \
--job-name=$ACCOUNT:inference \
--partition=batch \
--time=01:00:00 \
--container-image $CONTAINER_IMAGE \
--container-mounts $MLM_PATH:/workspace/megatron-lm/,$GPT_CKPT:/workspace/mcore_gpt_ckpt,$VOCAB_MERGE_FILE_PATH:/workspace/tokenizer \
--no-container-mount-home \
--pty /bin/bash \

# Inside the container run the following. 

cd megatron-lm/
export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_ARGS=(
    --vocab-file /workspace/tokenizer/gpt2-vocab.json
    --merge-file /workspace/tokenizer/gpt2-merges.txt
    --tokenizer-type GPT2BPETokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --load /workspace/mcore_gpt_ckpt
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-tokens-to-generate 20
    --max-batch-size 4
)

torchrun --nproc-per-node=4 examples/inference/gpt/simple_gpt_batch_inference.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
    --prompts "prompt one " "sample prompt two" "sample prompt 3"

NOTE: Other parameters which can be customized for inference are :-
--temperature (Sampling temperature)
--top_k (top_k sampling)
--top_p (top_p sampling)
--num-tokens-to-generate (Number of tokens to generate for each prompt)
--inference-batch-times-seqlen-threshold (During inference, if batch-size times sequence-length is smaller than this threshold then we will not use pipelining, otherwise we will.')
--use-dist-ckpt (If using dist checkpoint format for the model)
--use-legacy-models (If using legacy gpt model instead of mcore gpt model)

```


<br>


#### 2. Control Flow in the MCore Backend
An example of inference with static batching is provided in [gpt_batch_inference.py](./gpt/gpt_batch_inference.py).
* [mcore_engine](../../megatron/core/inference/engines/mcore_engine.py) **generate()** function is called with the input prompts.
* The `Scheduler` in the engine will add these prompts to the [active requests] pool (../../megatron/core/inference/inference_request.py) until max batch size is hit. Remaining requests will be added to the waiting requests pool. 
* The engine will run until all requests (waiting + active) are completed. 
    * The active requests are passed into  **generate_all_output_tokens_static_batch()** of the text generation controller . 
    * This function uses the **prep_model_for_inference()** method of the [model_inference_wrappers](../../megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py) and runs an autoregressive sampling loop
    * In the autoregressive loop, the **get_batch_for_context_window()** method of the inference wrapper is called to slice out the input tokens and masks
    * Input tokens and masks are passed it into the **run_one_forward_step()** method, which calls the model `.forward()` method to get the output logits
    * Output logits are synchronized across all pipeline parallel ranks
    * The text generation controller obtains the log probabilities and samples tokens based on the strategy defined in the sampling parameters.
    * The sampled tokens are then appended to the input prompt tokens for the next iteration 
    * The **update_generation_status()** method of the text generation controller checks which prompts have finished generating or hit a stop condition
    * After the inference loop, the result is detokenized and stored as an attribute of the InferenceRequest. These requests are marked as completed. 
    * The **update_requests_pool()** method of the scheduler moves completed requests into the completed request pool and waiting requests into the active request pool

<br>

#### 3. Customizing The Inference Pipeline

The inference pipeline supports three levels of customization:

* **Inference engine** - The MCore Engine is currently supported. Change this to add a new backend.
* **Text generation controller** - The main sampling loop. This can be customized to support alternative tokenization, detokenization, or to implement a new sampling strategy.
* **Inference Wrapped Model** - Change this to support a new model.
* **Modify Inference Parameters** - Change this to update top_p, top_k, number of tokens to be generated, temperature, or other sampling parameters.

<br>

##### 3.1. Create Your Own Inference Backend 
The  [abstract_engine.py](./../../megatron/core/inference/engine/abstract_engine.py) file contains a `generate` method that can be extended to support a new backend. 

```python
class AbstractEngine(ABC):
    @staticmethod
    def generate(self) -> dict:
        """The abstract backend's generate function. 

        To define a new backend, implement this method and return the outputs as a dictionary. 
```

<br>

##### 3.2. Implement a new Sampling Loop 

The [TextGenerationController](../../megatron/core/inference/text_generation_controllers/text_generation_controller.py) contains the main sampling loop and can be modified to support new tokenization, detokenization, or sampling strategies.

``` python
class TextGenerationController:

    def tokenize_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to tokenize the input prompts"""

    def sample_from_logits(
        self,
        last_token_logits: torch.Tensor,
        sampling_params: SamplingParams,
        vocab_size: int,
    ) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples according to the parameters defined in sampling_params and returns the sampled tokens.
        """

    def update_generation_status(
        self,
        updated_prompts_tokens: torch.Tensor,
        generation_started: torch.Tensor,
        current_context_end_position: int,
        is_generation_done_tensor: torch.Tensor,
        generated_sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Function to check which prompts have reached an end condition

        We check which prompts have reached an end condition and set the corresponding flags of the is_generation_done_tensor to True . The generated sequence lengths increases as we keep generating, until that prompts hits an eod condition. The generation started status tensor helps us determine which prompts have started generating
        """

    def generate_all_output_tokens_static_batch(
        self, active_requests: OrderedDict[int, InferenceRequest],
    ) -> OrderedDict[int, InferenceRequest]:
        """Utility to generate all the output tokens and probabilities for the prompts .

        This utility generates the output tokens for a static batch. It runs the forward steps till all prompts complete generation, updates the status of these requests to completed, adds the generated result and returns these requests
        """

    def detokenize_generations(self, prompt_tokens_with_generated_tokens: torch.Tensor) -> str:
        """Detokenize the output generations"""
```

<br>

##### 3.3. Support Other Models
Extend [abstract_model_inference_wrapper.py](./../../megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py) to support other models. The abstract model wrapper implements: 
* Forward method which calls the model `forward` method depending on model parallel settings
* Initializes the model and puts it in `.eval()` mode
* Setup for the input parameters (max batch size, max seq length) 

The following methods should be implemented: 
```python
class AbstractModelInferenceWrapper:
    def prep_model_for_inference(self, prompts_tokens: torch.Tensor):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop. It puts the model in eval mode , and gets some model and inference data parameters. Extend this to build position ids ,attention mask etc, so that required slices can be extracted during the forward pass
        """

    @abc.abstractclassmethod
    def get_batch_for_context_window(self) -> List:
        """Returns the input data for inference 

        This function gets called iteratively in the inference loop. It can be used to extract relevant input from the prompt tokens, attention mask etc. required for each step in inference.
```

Refer to [gpt_inference_wrapper.py](../../megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py) for an example of implementing this for GPTModel.

<br>

##### 3.3. Modify Inference Parameters
We use  [common inference params](../../megatron/core/inference/sampling_params.py) for text generation. Customize this if you want to change top_p, top_k, number of tokens to generate etc. If you want to add other attributes that you would use in the inference loop, you can do that as shown below

```
from megatron.core.inference.sampling_params import SamplingParams

c = SamplingParams(temperature=0.5)
c.add_attributes({'min_length':4, 'eod_id':153})
```

<br>

#### 4. Future work
The following features are planned for the future releases. 
* Dynamic batching 
* Paged Attention
* TRTLLM Engine support
* Support for multimodal inference