# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
        tokens_to_generate = int(input("Enter number of tokens to generate: "))
        data = json.dumps({"prompts": [sentence], "tokens_to_generate":tokens_to_generate})
        req = PutRequest(url, data, {'Content-Type': 'application/json'})
        response = urllib2.urlopen(req)
        resp_sentences = json.load(response)
        print("Megatron Response: ")
        print(resp_sentences["text"][0])
