import requests
import json
from human_eval.data import write_jsonl, read_problems


NUM_SAMPLES_PER_TASK = 1
stop_tokens = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<filename>", "<file_sep>", "<|endoftext|>"]


def query_server(prompt):
    url = 'http://localhost:8080/api'
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"prompts": [prompt], "tokens_to_generate": 512}
    response = requests.put(url, json=data, headers=headers)
    result = json.loads(response.text)["text"]
    return result[0]


def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def postprocess_generation(generation, prompt):
    """Defines the postprocessing for a LM generation.
    :param generation: str
        code generation from LM
    :param idx: int
        (not used for Humaneval-Task)
    """
    if not generation.startswith(prompt[:20]):
        print(f"issue with generation: {generation}")
        print(f"origin prompt: {prompt}")
    generation = generation[len(prompt) :]
    return prompt + stop_at_stop_token(generation, stop_tokens)


def main():
    problems = read_problems()
    prompts = [
                problems[task_id]["prompt"]
                for task_id in problems
                for _ in range(NUM_SAMPLES_PER_TASK)
            ]

    errors = []
    success = 0
    generations = []
    postprocessed_generations = []
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()  
        try:
            result = query_server(prompt)
            generations.append([result])
            postprocessed_generations.append([postprocess_generation(result, prompt)])
            success += 1
        except Exception as e:
            print(f"Error processing problem '{i}': {e}")
            errors.append(i)
        if i % 10 == 0:
            print(f"Processed {i} problems")
            print(f"Failed problem generations are: {errors}")
            #print(f"Example:\n{result}END\n")

    print(f"Done! {success} successful problems out of {len(prompts)}, failed are: {errors}")

    with open('megatron_generations.json', 'w') as f:
        json.dump(generations, f)

    with open('megatron_postprocessed_generations.json', 'w') as f:
        json.dump(postprocessed_generations, f)
    

if __name__ == '__main__':
    main()
