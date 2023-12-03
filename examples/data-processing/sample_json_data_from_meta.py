import os
import json
import argparse

def load_jsonl_file(output_folder, file_path, meta_keys):
    fileptr_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                dt = json.loads(line)
            except json.JSONDecodeError:
                pass  # Or handle the error as you see fit
            try:
                meta_idx = dt
                for meta_key in meta_keys:
                    meta_idx = meta_idx[meta_key]
                if meta_idx not in fileptr_dict:
                    fileptr_dict[meta_idx] = open(f"{output_folder}/{meta_idx}.jsonl", "w")
                fileptr_dict[meta_idx].write(f"{json.dumps(dt)}\n")
            except:
                pass

def process_files(output_folder_path, file_paths, meta_keys):
    for file_path in file_paths:
        try:
            load_jsonl_file(output_folder_path, file_path, meta_keys)
        except Exception as exc:
            print(f'{file_path} generated an exception: {exc}')
            raise

def main():
    parser = argparse.ArgumentParser(description='Count words in a JSONL file.')
    parser.add_argument('--input-folder-path', type=str, help='Path to the input folder')
    parser.add_argument('--output-folder-path', type=str, help='Path to the output folder')
    parser.add_argument('--meta-keys', nargs='+', type=str, help='Meta keys to split on')
    args = parser.parse_args()

    files = os.listdir(args.input_folder_path)
    file_paths = [os.path.join(args.input_folder_path, f) for f in files if f.endswith("jsonl")]
    process_files(args.output_folder_path, file_paths, args.meta_keys)

if __name__ == "__main__":
    main()
