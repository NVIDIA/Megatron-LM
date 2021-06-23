# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from megatron.text_generation_utils import pad_batch
from megatron.text_generation_utils import get_token_stream2

GENERATE_NUM = 0

def tokenize_batch(sentences):
    args = get_args()
    tokenizer = get_tokenizer()
    context_tokens = [tokenizer.tokenize(s) for s in sentences]
    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor 


class MegatronGenerate(Resource):
    def __init__(self, model):
        self.model = model
    
    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
    
    @staticmethod
    def send_generate_info(context_tokens_tensor, context_length_tensor, max_len):
        """
        Needs to be synced up with receive_generate_info
        """
        # Send the sizes of the tensors
        input_info = [context_tokens_tensor.size(0), context_tokens_tensor.size(1), max_len]
        input_info_tensor = torch.cuda.LongTensor(input_info)
        torch.distributed.broadcast(input_info_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

        # Now send tensors
        torch.distributed.broadcast(context_length_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        torch.distributed.broadcast(context_tokens_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    @staticmethod
    def receive_generate_info():
        """
        Needs to be synced up with send_generate_info
        """
        input_info_tensor = torch.empty(3, dtype=torch.int64, device=torch.device("cuda"))
        torch.distributed.broadcast(input_info_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        batch_size = input_info_tensor[0].item()
        seq_len = input_info_tensor[1].item()
        max_len = input_info_tensor[2].item()
        
        context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.device("cuda"))
        context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.device("cuda"))
        
        torch.distributed.broadcast(context_length_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        torch.distributed.broadcast(context_tokens_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        return context_length_tensor, context_tokens_tensor, max_len
    
    @staticmethod
    def do_generate(model, context_length_tensor, context_tokens_tensor, max_len):
        token_stream = get_token_stream2(model, context_tokens_tensor, context_length_tensor)
        for i, decode_tokens in enumerate(token_stream):
            if i == max_len-1:
                break
            pass
        return decode_tokens
    
    def put(self):
        sentences = request.get_json()["sentences"]
        max_len = 1024  # TODO (rprenger) this should not be hardcoded
        if "max_len" in request.get_json():
            max_len = request.get_json()["max_len"]

        context_tokens_tensor, context_length_tensor = tokenize_batch(sentences)
        MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
        MegatronGenerate.send_generate_info(context_tokens_tensor, context_length_tensor, max_len)  # Send them info
        decode_tokens = MegatronGenerate.do_generate(self.model, context_length_tensor, context_tokens_tensor, max_len)  # Do stuff
        
        args = get_args()
        tokenizer = get_tokenizer()
        decode_tokens, _ = decode_tokens
        decode_tokens = decode_tokens[0].cpu().numpy().tolist()
        trim_decode_tokens = tokenizer.detokenize(decode_tokens)
        return jsonify({"sentences": [trim_decode_tokens]})
    

class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__)
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/generate', resource_class_args=[model])

    def run(self, url):
        self.app.run(url, debug=False)
