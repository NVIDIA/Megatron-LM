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
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api

from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from megatron.text_generation_utils import tokenize_batch, get_token_stream

GENERATE_NUM = 0

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
        torch.distributed.broadcast(input_info_tensor, 0)

        # Send variables to all ranks 
        torch.distributed.broadcast(context_length_tensor, 0)
        torch.distributed.broadcast(context_tokens_tensor, 0)

    @staticmethod
    def receive_generate_info():
        """
        Needs to be synced up with send_generate_info
        """
        input_info_tensor = torch.empty(3, dtype=torch.int64, device=torch.device("cuda"))
        torch.distributed.broadcast(input_info_tensor, 0)
        batch_size = input_info_tensor[0].item()
        seq_len = input_info_tensor[1].item()
        max_len = input_info_tensor[2].item()
        
        context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.device("cuda"))
        context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.device("cuda"))
        
        # Send variables to all ranks 
        torch.distributed.broadcast(context_length_tensor, 0)
        torch.distributed.broadcast(context_tokens_tensor, 0)
        
        return context_length_tensor, context_tokens_tensor, max_len
    
    @staticmethod
    def do_generate(model, context_length_tensor, context_tokens_tensor, max_len):
        token_stream = get_token_stream(model, context_tokens_tensor, context_length_tensor)
        for i, decode_tokens in enumerate(token_stream):
            if i == max_len-1:
                break
            pass
        return decode_tokens
    
    def put(self):
        args = get_args()
        sentences = request.get_json()["sentences"]
        if len(sentences) > 128:
            return "Maximum number of sentences is 128", 400

        max_len = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "max_len" in request.get_json():
            input_max_len = request.get_json()["max_len"]
            if input_max_len < args.seq_length:
                max_len = input_max_len

        context_tokens_tensor, context_length_tensor = tokenize_batch(sentences)
        MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
        MegatronGenerate.send_generate_info(context_tokens_tensor, context_length_tensor, max_len)  # Send them info
        decode_tokens = MegatronGenerate.do_generate(self.model, context_length_tensor, context_tokens_tensor, max_len)  # Do stuff
        args = get_args()
        tokenizer = get_tokenizer()
        decode_tokens, _ = decode_tokens
        resp_sentences = []
        for i in range(decode_tokens.size(0)):
            decode_token = decode_tokens[i,:].cpu().numpy().tolist()
            resp_sentences.append(tokenizer.detokenize(decode_token))
        return jsonify({"sentences": resp_sentences})
    

def index():
    return current_app.send_static_file('index.html')

class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', index)
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/generate', resource_class_args=[model])

    def run(self, url):
        self.app.run(url, debug=False)
