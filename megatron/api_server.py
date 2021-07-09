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
from megatron import mpu
from megatron.text_generation_utils import generate

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

        MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
        resp_sentences = generate(self.model, sentences, max_len) 
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
