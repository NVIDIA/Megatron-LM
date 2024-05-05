# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests


if __name__ == "__main__":
    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}

    while True:
        sentence = input("Enter prompt: ")
        image_path = input("Enter image_path (leave blank if no image): ")
        tokens_to_generate = int(eval(input("Enter number of tokens to generate: ")))

        image_dict = {} if not image_path else {"image_path": image_path}

        data = {**image_dict, "prompts": ["[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n{} [/INST]\n".format(sentence)], "tokens_to_generate": tokens_to_generate, "temperature": 0.8, "top_k": 1}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
