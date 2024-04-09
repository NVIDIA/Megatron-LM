### Megatron Core Inference Documentation
This guide will walk you through how you can use megatron core for inference on your models. 

### Contents
- [Megatron Core Inference Documentation](#megatron-core-inference-documentation)
- [Contents](#contents)
  - [1. Quick Start](#1-quick-start)
    - [1.1 Understanding The Code](#11-understanding-the-code)
    - [1.2 Running The Code](#12-running-the-code)
  - [2. A More Involved Example](#2-a-more-involved-example)
  - [3. Flow of Control In MCore Backend](#3-flow-of-control-in-mcore-backend)
  - [4. Customizing The Inference Pipeline](#4-customizing-the-inference-pipeline)
    - [4.1. Create Your Own Inference Backend](#41-create-your-own-inference-backend)
    - [4.2. Create Your Own Text Generation Strategy](#42-create-your-own-text-generation-strategy)
    - [4.3. Support Other Models](#43-support-other-models)

<br>

#### 1. Quick Start
This will walk you through the flow of running inference on a GPT model trained using megatron core. The file can be found at [quick_start.py](./quick_start.py)

<br>

##### 1.1 Understanding The Code
***STEP 1 - We initalize model parallel and other default aruguments***
We can default micro batch size to be 1, since for TP models its not used, and for PP models it is calculated during runtime. 
```python
    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True, 'micro_batch_size': 1}
    )
```

***STEP 2 - We load the model using the model_provider_function***
NOTE: The model provider function in the quickstart just supports mcore model. Check [generate_mcore_samples_gpt.py](./gpt/generate_mcore_samples_gpt.py) to see how to support megatorn lm models as well.
```python
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]
```

***STEP 3 - Choose a backend***
One of the important elements of the generate function is a backend. In this example we will be choosing the [megatorn core backend](../../megatron/core/inference/backends/mcore_backend.py) with a [simple text generation strategy](../../megatron/core/inference/text_generation_strategies/simple_text_generation_strategy.py). (Other backends that will be supported are [TRTLLMBackend](../../megatron/core/inference/backends/trt_llm_backend.py)). If you dont want any customization use mcore backend with simple text generation strategy.
```python
    inference_wrapped_model = GPTInferenceWrapper(model, args)
    text_generation_strategy = SimpleTextGenerationStrategy(
        inference_wrapped_model=inference_wrapped_model, 
        tokenizer=tokenizer
    )
    inference_backend = MCoreBackend(
        text_generation_strategy=text_generation_strategy
    )
```

***STEP 4 - Run the generate function and display results***
We use default values for the [common inference params](../../megatron/core/inference/common_inference_params.py). Customize this if you want to change top_p, top_k, number of tokens to generate etc. 
*Note that the result is returned as a dictionary only on rank 0.*
```python
    result = common_generate(
        inference_backend=inference_backend,
        prompts=["How large is the universe ?", "Where can you celebrate birthdays ? "],
        common_inference_params=CommonInferenceParams(),
    )

    if torch.distributed.get_rank() == 0:
        print(result['prompts_plus_generations_detokenized'])
```

<br>

##### 1.2 Running The Code
An example of running the file is shown below. Change TP,PP values, model spec , tokenizer etc according to your model . 

*NOTE: Most of these can be obtained from the script you used to train the model*
```

TOKENIZER_ARGS=(
    --vocab-file /workspace/megatron-lm/gpt2-vocab.json
    --merge-file /workspace/megatron-lm/gpt2-merges.txt
    --tokenizer-type GPT2BPETokenizer
)

MODEL_PARALLEL_ARGS=(
   --tensor-model-parallel-size 2 
   --pipeline-model-parallel-size 2
)

MODEL_SPEC=(
    --num-layers 8 
    --hidden-size 256 
    --num-attention-heads 8 
    --seq-length 512 
    --max-position-embeddings 512 
    --use-mcore-models 
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
)
torchrun --nproc-per-node=4 examples/inference/quick_start.py \
    --load /workspace/checkpoint/tp2pp2 \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MODEL_SPEC[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
```

<br>

#### 2. A More Involved Example
The example in [generate_mcore_samples_gpt.py](./gpt/generate_mcore_samples_gpt.py) is more involved. It shows you the following
* Loading mcore/megatron lm checkpoint
* Customizing inference parameters using command line aruguments
* Reading prompts in batches from a file and writing results to a file

<br>  

#### 3. Flow of Control In MCore Backend
The following is what happens in the [generate_mcore_samples_gpt.py](./gpt/generate_mcore_samples_gpt.py) text generation part.
* We call the [common_generate_function](../../megatron/core/inference/common_generate_function.py) with the megatron core backend and the list of input prompts and inference parameters
* This in turn calls the [mcore_backend](../../megatron/core/inference/backends/mcore_backend.py) **generate()** function. 
* This function uses the [simple_text_generation_strategy](../../megatron/core/inference/text_generation_strategies/simple_text_generation_strategy.py) to pad and tokenize input prompts 
* The padded prompts are passed into the **generate_output_tokens()** of the text generation strategy . 
* This function uses the [model_inference_wrappers](../../megatron/core/inference/inference_model_wrappers/abstract_model_inference_wrapper.py) **prep_model_for_inference()** , and then runs an auto regressive loop
* In the auto regressive loop the inference wrappers **get_batch_for_context_window()** is called to get the required input, which is passed into the __call__ method, which takes care of calling the appropriate (PP, TP) model forward methods to get the output logits
* The text generation strategy then samples from these logits and obtains the log probabilities based on the common inference parameters.
* The input prompt tokens are updated with the results and then copied from last stage to first stage in case of PP models.  
* The **update_generation_status** of the text generation strategy is called to check which of the prompts have completed generating , what the generation lengths are etc. 
* The status of the prompts generations is broacasted so that in case of early stopping all ranks can break. 
* Finally after the inference loop, the tokens are passed to the text generation strategies *detokenize_generations()* function to get the generated text . 

<br>

#### 4. Customizing The Inference Pipeline
The following guide will walk you through how you can customize different parts of the inference pipeline. Broadly there are three levels at which you can customize the pipeline. 
* **Inference backend** - Highest level of customization. (Currently we support MCore and TRTLLM backends). Change this if you completely want to add your own way of running inference.  
* **Text generation strategy** - Extend this if you want to customize tokenization, text generation or detokenization
* **Inference Wrapped Model** - Change this if you just want to support a new model 

<br>

##### 4.1. Create Your Own Inference Backend 
This is the highest level of customization. The  [abstract_backend.py](./../../megatron/core/inference/backends/abstract_backend.py) file has a core generate method that you can extend to support your own backend. 

```python
class AbstractBackend(ABC):
    @staticmethod
    def generate(self) -> dict:
        """The abstarct backends generate function. 

        To define your own backend, make sure you implement this and return the outputs as a dictionary . 
```

Currently we support mcore backend. Soon we will suport TRT-LLM. The suggested flow as you can see from the [generate_mcore_samples_gpt.py](./gpt/generate_mcore_samples_gpt.py) is to choose TRTLLM Backend as a default, and if the model fails the export, we will use the megatron core backend. 


<br>

##### 4.2. Create Your Own Text Generation Strategy
In case you want to use the megatron core backend, but would like to overwrite the tokenization, text generation or detokenization extend the [simple_text_generation_strategy.py](../../megatron/core/inference/text_generation_strategies/simple_text_generation_strategy.py). The class has the following methods
``` python
class SimpleTextGenerationStrategy:

    def tokenize_and_pad_input_prompts(
            self, prompts: List[str], num_tokens_to_generate: int
        ) -> Tuple[torch.Tensor, torch.Tensor]
        """Utility to tokenize and pad the input prompts

            Tokenizes the input prompts, pads them to required length and returns the tokenized tensor and also the original prompt lengths.
        """

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
        updated_promps_tokens: torch.Tensor,
        generation_started: torch.Tensor,
        current_context_end_position: int,
        is_generation_done_tensor: torch.Tensor,
        actual_plus_generated_sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Function to check which prompts have reached an end condition

        We check which prompts have reached an end condition and set the corresponding flags of the is_generation_done_tensor to True . The generated sequence lengths starts off with input prompt lengths values and increases as we keep generating, until that prompts hits an eod condition. The generation started status tensor helps us determine which are generated tokens, and which are input prompt tokens
        """

    def generate_output_tokens(
        self,
        prompts_tokens: torch.Tensor,
        prompts_lengths: torch.Tensor,
        common_inference_params: CommonInferenceParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility to generate the output tokens and probabilities for the prompts

        This utility generates the output tokens. It uses the model wrapper to generate the outputs internally
        """

    def detokenize_generations(
        self, prompt_tokens_with_generations: torch.Tensor, required_sequence_lengths: torch.Tensor
    ) -> List[str]:
        """Detokenize the output generations

        This function takes the prompts with the generated tokens, and detokenizes it and trims off according to the generated sequence length param
        """
```

<br>

##### 4.3. Support Other Models
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