### Megatron Core Inference Documentation
This guide will walk you through how you can use megatron core for inference on your models. 

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
This will walk you through the flow of running batch inference on a GPT model trained using megatron core. The file can be found at [simple_gpt_batch_inference.py](./gpt/simple_gpt_batch_inference.py)

<br>

##### 1.1 Understanding The Code
***STEP 1 - We initalize model parallel and other default aruguments***
We can default micro batch size to be 1, since for TP models it is not used, and for PP models it is calculated during runtime. 
```python
    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True, 'micro_batch_size': 1}
    )
```

***STEP 2 - We load the model using the model_provider_function***
NOTE: The model provider function in the script supports MCore and Legacy models. 

```python
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]
```

***STEP 3 - Choose an engine***
One of the important elements of the generate function is an inference engine. In this example we will be choosing the [megatorn core enge](../../megatron/core/inference/engine/mcore_engine.py) with a [simple text generation controller](../../megatron/core/inference/text_generation_controllers/simple_text_generation_controller.py) since TRTLLMEngine is not available yet. Other engines that will be supported are [TRTLLMEngine](../../megatron/core/inference/engine/trt_llm_engine_wrapper.py)). If you dont want any customization use mcore engine with simple text generation controller.
```python
    inference_wrapped_model = GPTInferenceWrapper(model, args)
    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, 
        tokenizer=tokenizer
    )
    inference_backend = MCoreEngine(
        text_generation_controller=text_generation_controller, max_batch_size=args.max_batch_size
    )
```

***STEP 4 - Run the generate function and display results***
We use default values for the [common inference params](../../megatron/core/inference/common_inference_params.py). Customize this if you want to change top_p, top_k, number of tokens to generate etc. 
*Note that the result is returned as a list of [InferenceRequests](../../megatron/core/inference/inference_request.py)*
```python
    results: List[InferenceRequest] = inference_engine.generate(
        prompts=args.prompts, common_inference_params=common_inference_params
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
An example of running the file is shown below. Change tokenizer paths, inference params etc.for your model . 

For a quick recap on inference params refer to [this blog](https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910) 

```

TOKENIZER_ARGS=(
    --vocab-file /workspace/megatron-lm/gpt2-vocab.json
    --merge-file /workspace/megatron-lm/gpt2-merges.txt
    --tokenizer-type GPT2BPETokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-tokens-to-generate 20
    --max-batch-size 4
)

torchrun --nproc-per-node=4 examples/inference/gpt/simple_gpt_batch_inference.py \
    --load /workspace/checkpoint/tp2pp2 \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} 
    --prompts "prompt one " "sample prompt two" "sample prompt 3"

NOTE: Other parameters which can be customized for inference are :-
--temperature (Sampling temperature)
--top_k (top_k sampling)
--top_p (top_p sampling)
--num-tokens-to-generate (Number of tokens to generate for each prompt)
--inference-batch-times-seqlen-threshold (During inference, if batch-size times sequence-length is smaller than this threshold then we will not use pipelining, otherwise we will.')

```


<br>


#### 2. Flow of Control In MCore Backend
The following is what happens in the [simple_gpt_batch_inference.py](./gpt/simple_gpt_batch_inference.py) text generation part.
* We call  [mcore_engine](../../megatron/core/inference/engine/mcore_engine.py) **generate()** function with all our input prompts.
* The scheduler in the engine will add these prompts to [active requests](../../megatron/core/inference/inference_request.py) till we hit max batch size, and then it will put the rest in waiting requests. 
* The engine will then run till all requests (waiting + active) are completed 
    * The active requests are passed into  **generate_all_output_tokens_static_batch()** of the text generation controller . 
    * This function uses the [model_inference_wrappers](../../megatron/core/inference/inference_model_wrappers/abstract_model_inference_wrapper.py) **prep_model_for_inference()** , and then runs an auto regressive loop
    * In the auto regressive loop the inference wrappers **get_batch_for_context_window()** is called to get the required input, which is passed into the **run_one_forward_step()** method, which takes care of calling the appropriate (PP, TP) model forward methods to get the output logits
    * The output logits are synchronized across all ranks for PP Models
    * The text generation controller obtains the log probabilities and samples tokens based on the common inference parameters.
    * The sampled tokens are then appended to the input prompt tokens for the next iteration 
    * The **update_generation_status()** of the text generation controller is called to check which of the prompts have completed generating , what the generation lengths are etc. 
    * Finally after the inference loop, the result is detokenized and stored back into the inference requests. The status of these requests are marked as completed. 
    * We then use the schedulers **update_requests_pool()** to update the requests pools. (i.e) Completed requests are put into the completed request pool and the waiting requests are added into the active request pool

<br>

#### 3. Customizing The Inference Pipeline
The following guide will walk you through how you can customize different parts of the inference pipeline. Broadly there are three levels at which you can customize the pipeline. 
* **Inference engine** - Highest level of customization. (Currently we support MCore Engine). Change this if you completely want to add your own way of running inference.  
* **Text generation controller** - Extend this if you want to customize tokenization, text generation, sampling, detokenization etc.
* **Inference Wrapped Model** - Change this if you just want to support a new model 
* **Modify Inference Parameters** - Change this to update top_p, top_k, number of tokens to be generated, temperature etc.

<br>

##### 3.1. Create Your Own Inference Backend 
This is the highest level of customization. The  [abstract_engine.py](./../../megatron/core/inference/engine/abstract_engine.py) file has a core generate method that you can extend to support your own backend. 

```python
class AbstractEngine(ABC):
    @staticmethod
    def generate(self) -> dict:
        """The abstarct backends generate function. 

        To define your own backend, make sure you implement this and return the outputs as a dictionary . 
```

Currently we support mcore engine. Soon we will suport TRT-LLM. The suggested flow as you can see from the [simple_gpt_batch_inference.py](./gpt/simple_gpt_batch_inference.py) is to choose TRTLLM Backend as a default, and if the model fails the export, we will use the megatron core backend. 


<br>

##### 3.2. Create Your Own Text Generation Controller
In case you want to use the megatron core backend, but would like to overwrite the tokenization, text generation or detokenization extend the [simple_text_generation_controller.py](../../megatron/core/inference/text_generation_controllers/simple_text_generation_controller.py). The class has the following methods
``` python
class SimpleTextGenerationController:

    def tokenize_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to tokenize the input prompts"""

    def sample_from_logits(
        self,
        last_token_logits: torch.Tensor,
        common_inference_params: CommonInferenceParams,
        vocab_size: int,
    ) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it according to the parameters defined in common_inference_params and returns the samples
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
In order to support other models please extend the [abstract_model_inference_wrapper.py](./../../megatron/core/inference/inference_model_wrappers/abstract_model_inference_wrapper.py) file. The abstract wrapper already supports the following :
* Forward method which automatically calls the appropriate forward method (PP or TP etc) depending on model parallel settings
* Initalizes the model and puts it in eval mode
* Obtains the input parameters (batch size, max seq length) and has an instance of the input 

The main methods to change for your model might be the following: 
```python
class AbstractModelInferenceWrapper:
    def prep_model_for_inference(self, prompts_tokens: torch.Tensor):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop. It puts the model in eval mode , and gets some model and inference data parameters. Extend this to build position ids ,attention mask etc, so that required slices can be extracted during the forward pass
        """

    @abc.abstractclassmethod
    def get_batch_for_context_window(self) -> List:
        """Returns the input data for inference 

        This function gets called iteratively in the inference loop . It can be used to extract relevant input from the prompt tokens, attention mask etc. required for each step in inference.
```

To see an example of how we extend this for gpt please refer [gpt_inference_wrapper.py](../../megatron/core/inference/inference_model_wrappers/gpt/gpt_inference_wrapper.py)

<br>

##### 3.3. Modify Inference Parameters
We use  [common inference params](../../megatron/core/inference/common_inference_params.py) for text generation. Customize this if you want to change top_p, top_k, number of tokens to generate etc. If you want to add other attributes that you would use in the inference loop, you can do that as shown below

```
from megatron.core.inference.common_inference_params import CommonInferenceParams

c = CommonInferenceParams(temperature=0.5)
c.add_attributes({'min_length':4, 'eod_id':153})
```

<br>

#### 4. Future work
The following are planned for the future releases . 
* Dynamic batching 
* Paged Attention
* TRTLLM Engine support
* Support for Multimodal model inference