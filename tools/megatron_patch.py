# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoTokenizer

def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

_GLOBAL_TOKENIZER = None

def get_tokenizer():
    """Return tokenizer."""
    return _GLOBAL_TOKENIZER

def build_tokenizer(args):

    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.patch_tokenizer_type))
    # Select and instantiate the tokenizer.
    if args.patch_tokenizer_type == 'JiebaBPETokenizer':
        from .jiebabpe_tokenizer import JiebaBPETokenizer
        tokenizer = JiebaBPETokenizer(args.patch_vocab_file)
        args.padded_vocab_size = _vocab_size_with_padding(
            tokenizer.vocab_size, args)
    elif args.patch_tokenizer_type == 'BloomTokenizerFromHF':
        from transformers import BloomTokenizerFast as BloomTokenizer
        if args.load is None:
            tokenizer = BloomTokenizer.from_pretrained('bigscience/bloom-560m')
        else:
            tokenizer = BloomTokenizer.from_pretrained(args.load)
        args.padded_vocab_size = 250880
    elif args.patch_tokenizer_type == 'ChatGLMTokenizerFromHF':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 130528
    elif args.patch_tokenizer_type == 'GLM10BZHTokenizerFromHF':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-10b-chinese',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 50048
    elif args.patch_tokenizer_type == 'IcetkGLM130BTokenizer':
        from .icetk_glm130b_tokenizer import _IceTokenizer
        tokenizer = _IceTokenizer()
        args.padded_vocab_size = 150528
    elif args.patch_tokenizer_type == 'OPTTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side='right',
            use_fast=False,
        )
        DEFAULT_PAD_TOKEN = '<pad>'
        DEFAULT_EOS_TOKEN = '</s>'
        DEFAULT_BOS_TOKEN = '<s>'
        DEFAULT_UNK_TOKEN = '<unk>'

        special_tokens_dict = dict()
        if not tokenizer.pad_token:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        if not tokenizer.eos_token:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if not tokenizer.bos_token:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if not tokenizer.unk_token:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'LLamaTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<unk>"))

        tokenizer.eod = tokenizer.eos_token_id
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'FalconTokenizer':
        if args.load is None:
            tokenizer = AutoTokenizer.from_pretrained(
                'tiiuae/falcon-7b',
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.load,
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size
        tokenizer.pad_token = tokenizer.eos_token
    elif args.patch_tokenizer_type == 'BaichuanTokenizer':
        from .tokenization_baichuan import BaichuanTokenizer
        if args.load is None:
            tokenizer = BaichuanTokenizer.from_pretrained(
                'baichuan-inc/Baichuan-13B-Base',
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        else:
            tokenizer = BaichuanTokenizer.from_pretrained(
                args.load,
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        DEFAULT_PAD_TOKEN = '[PAD]'
        DEFAULT_EOS_TOKEN = '</s>'
        DEFAULT_BOS_TOKEN = '<s>'
        DEFAULT_UNK_TOKEN = '<unk>'

        special_tokens_dict = dict()
        if not tokenizer.pad_token:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        if not tokenizer.eos_token:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if not tokenizer.bos_token:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if not tokenizer.unk_token:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'QwenTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
        if hasattr(tokenizer, 'eod_id'):
            tokenizer.eos_token_id = tokenizer.eod_id
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'Qwen2Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _Qwen2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))

                if self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                    try:
                        test_conversation = [
                            {'role': 'user', 'content': 'hello world'}
                        ]
                        self.apply_chat_template(test_conversation)
                    except Exception:
                        # the default chat_template is invalid, assume user will not do SFT
                        self.tokenizer.chat_template = None 
                
            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations):
                return self.tokenizer.apply_chat_template(conversations)
            
            @property
            def vocab_size(self):
                return len(self.tokenizer.encoder) + self.extra_vocab_size

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id


        tokenizer = _Qwen2Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

    elif args.patch_tokenizer_type == 'Qwen2VLTokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _Qwen2VLTokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                self.special_tokens_map = {k:v for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)}
                self.image_token = '<|image_pad|>'
                self.video_token = '<|video_pad|>'
                self.vision_start_token = '<|vision_start|>'
                self.vision_end_token = '<|vision_end|>'

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
                return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, **kwargs)
            
            @property
            def vocab_size(self):
                return len(self.tokenizer.encoder) + self.extra_vocab_size

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id
            
            @property
            def image_token_id(self):
                return self.special_tokens_map[self.image_token]
            
            @property
            def video_token_id(self):
                return self.special_tokens_map[self.video_token]
            
            @property
            def vision_start_token_id(self):
                return self.special_tokens_map[self.vision_start_token]
            
            @property
            def vision_end_token_id(self):
                return self.special_tokens_map[self.vision_end_token]
            
            def encode(self, x):
                return self.tokenizer.encode(x)

        tokenizer = _Qwen2VLTokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size


    elif args.patch_tokenizer_type == 'DeepSeekV2Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _DeepSeekV2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size

                if self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                    try:
                        test_conversation = [
                            {'role': 'user', 'content': 'hello world'}
                        ]
                        self.apply_chat_template(test_conversation)
                    except Exception:
                        # the default chat_template is invalid, assume user will not do SFT
                        self.tokenizer.chat_template = None

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
                return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, **kwargs)

            @property
            def vocab_size(self):
                return len(self.tokenizer) + self.extra_vocab_size - 2

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

        tokenizer = _DeepSeekV2Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

    elif args.patch_tokenizer_type == 'DeepSeekV3Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _DeepSeekV3Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size

                if self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                    try:
                        test_conversation = [
                            {'role': 'user', 'content': 'hello world'}
                        ]
                        self.apply_chat_template(test_conversation)
                    except Exception:
                        # the default chat_template is invalid, assume user will not do SFT
                        self.tokenizer.chat_template = None

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
                return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, **kwargs)

            @property
            def vocab_size(self):
                return len(self.tokenizer) + self.extra_vocab_size - 2

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

        tokenizer = _DeepSeekV3Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

    elif args.patch_tokenizer_type == 'QwenVLTokenizer':
        from .tokenization_qwen_vl import QWenTokenizer
        tokenizer = QWenTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
        tokenizer.eos_token_id = tokenizer.eod_id

        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'YiTokenizer':
        from .tokenization_yi import YiTokenizer
        if args.load is None:
            tokenizer = YiTokenizer.from_pretrained(
                '01-ai/Yi-6B',
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        else:
            tokenizer = YiTokenizer.from_pretrained(
                args.load,
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'MistralTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(args.load,
                                                  padding_side='right',
                                                  use_fast=False,)
        tokenizer.pad_token_id = 0
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'BloomTokenizerFromCustom':
        from transformers import BloomTokenizerFast as BloomTokenizer
        tokenizer = BloomTokenizer.from_pretrained(args.load)
        if 'mg' not in args.load:
            args.padded_vocab_size = 134298
        else:
            args.padded_vocab_size = _vocab_size_with_padding(
                tokenizer.vocab_size, args)
    elif args.patch_tokenizer_type == 'StarcoderTokenizerFromHF':
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        tokenizer.pad_token = tokenizer.eos_token
        args.padded_vocab_size = 49152

    elif args.patch_tokenizer_type == 'GPT2BPETokenizer':
        from megatron import get_tokenizer
        tokenizer = get_tokenizer()
        
    elif args.patch_tokenizer_type == 'LLama2Tokenizer' or args.patch_tokenizer_type == 'MixtralTokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _LLama2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<unk>"))

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                                      max_length=max_length, truncation=truncation,
                                      add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations):
                return self.tokenizer.apply_chat_template(conversations)

            @property
            def vocab_size(self):
                return self.tokenizer.vocab_size + self.extra_vocab_size

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

        tokenizer = _LLama2Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

    elif args.patch_tokenizer_type == 'LLama3Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _LLama3Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                # NOTE: Add pad token for LLaMA 3.1
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|finetune_right_pad_id|>"))
                
                if self.tokenizer.chat_template is None:
                    # Add a default template for LLaMA3.1
                    # from meta-llama-3.1-70b-instruct
                    self.tokenizer.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
                    try:
                        test_conversation = [
                            {'role': 'user', 'content': 'hello world'}
                        ]
                        self.apply_chat_template(test_conversation)
                    except Exception:
                        # the default chat_template is invalid, assume user will not do SFT
                        self.tokenizer.chat_template = None 
                

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations):
                return self.tokenizer.apply_chat_template(conversations)

            @property
            def vocab_size(self):
                return self.tokenizer.vocab_size + self.extra_vocab_size

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

        tokenizer = _LLama3Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

    elif args.patch_tokenizer_type == 'VicunaTokenizerFromHF':
        tokenizer = AutoTokenizer.from_pretrained(args.load,
                                                  model_max_length=args.seq_length,
                                                  padding_side="right",
                                                  use_fast=False)
        tokenizer.pad_token = tokenizer.unk_token
        args.padded_vocab_size = 32000

    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(
                                      args.patch_tokenizer_type))


    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER

