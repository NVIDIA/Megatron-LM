import json
import argparse
from concurrent.futures import ProcessPoolExecutor

def count_words_in_json_object(json_line):
    try:
        data = json.loads(json_line)
        text = data.get('text', '')
        return len(text.split())
    except json.JSONDecodeError:
        return 0

def run(file_path, workers=4):
    total_word_count = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(count_words_in_json_object, lines)
        total_word_count = sum(results)
    return total_word_count

def main():
    parser = argparse.ArgumentParser(description='Count words in a JSONL file.')
    parser.add_argument('--input-file', type=str, help='Path to the JSONL file')
    parser.add_argument('--num-proc', type=int, help='Path to the JSONL file')
    args = parser.parse_args()

    word_count = run(args.input_file, args.num_proc)
    print(f"{word_count}")

if __name__ == "__main__":
    main()
