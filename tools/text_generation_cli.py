# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import json
import sys
import urllib
from urllib.request import Request

if __name__ == "__main__":
    url = sys.argv[1]
    while True:
        sentence = input("Enter prompt: ")
        tokens_to_generate = int(input("Enter number of tokens to generate: "))
        data = json.dumps({"prompts": [sentence], "tokens_to_generate":tokens_to_generate})
        data = data.encode('utf-8')
        req = Request(url, data=data, headers={'Content-Type': 'application/json'}, method='PUT')
        response = urllib.request.urlopen(req)
        resp_sentences = json.load(response)
        print("Megatron Response: ")
        print(resp_sentences["text"][0])
