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
import json
import sys
import urllib2
class PutRequest(urllib2.Request):
    '''class to handling putting with urllib2'''

    def get_method(self, *args, **kwargs):
        return 'PUT'

if __name__ == "__main__":
    url = sys.argv[1]
    while True:
        sentence = raw_input("Enter prompt: ")
        max_len = int(input("Enter number tokens output: "))
        data = json.dumps({"sentences": [sentence], "max_len":max_len})
        req = PutRequest(url, data, {'Content-Type': 'application/json'})
        response = urllib2.urlopen(req)
        resp_sentences = json.load(response)
        print("Megatron Response: ")
        print(resp_sentences["sentences"][0])
