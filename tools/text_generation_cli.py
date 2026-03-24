# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import ast
import json
import sys

import requests


def _print_usage_and_exit() -> None:
    print("Usage: python tools/text_generation_cli.py <host:port>")
    sys.exit(1)


def _parse_tokens_to_generate(raw_value: str) -> int:
    try:
        tokens_to_generate = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        raise ValueError("Number of tokens to generate must be a valid integer.")

    if not isinstance(tokens_to_generate, int) or tokens_to_generate <= 0:
        raise ValueError("Number of tokens to generate must be a positive integer.")

    return tokens_to_generate


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _print_usage_and_exit()

    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}

    while True:
        sentence = input("Enter prompt: ")
        try:
            tokens_to_generate = _parse_tokens_to_generate(
                input("Enter number of tokens to generate: ")
            )
        except ValueError as exc:
            print(f"Input error: {exc}")
            continue

        data = {"prompts": [sentence], "tokens_to_generate": tokens_to_generate}
        try:
            response = requests.put(url, data=json.dumps(data), headers=headers, timeout=30)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            continue

        if response.status_code != 200:
            try:
                error_message = response.json().get('message', response.text)
            except ValueError:
                error_message = response.text
            print(f"Error {response.status_code}: {error_message}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
