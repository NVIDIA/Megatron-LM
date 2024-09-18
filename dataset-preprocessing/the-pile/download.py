from datasets import load_dataset
from datasets import disable_caching
from datasets import load_dataset_builder
from huggingface_hub import hf_hub_download
import os
import json
import time

save_path = "/N/scratch/jindjia/thepile"
dataset_output_name = 'pile.jsonl'

# disable_caching()
# --------convert it to json with local memory mode, you may at least have almost 1TB disk memory for c4 dataset ---------
# load dataset
# train_data = load_dataset('c4', 'en', split='train')
# columns_to_keep = ["text"]
# train_data = train_data.removecd_columns([col for col in train_data.column_names if col not in columns_to_keep])
# print(train_data.column_names)


# train_data.to_json(os.path.join(save_path, dataset_output_name),  lines=True)

# --------convert it to json with stream mode (if your disk is not of memory)---------
train_data = load_dataset('EleutherAI/the_pile_deduplicated', split='train', num_proc=16)

start_time = time.time()

# with open(os.path.join(save_path, dataset_output_name), 'w') as json_file:
#     for i, sample in enumerate(train_data):
#         json_record = json.dumps({"text": sample['text']})
#         json_file.write(json_record + '\n')
#         if i % 100000 == 0:
#             print(f'Processed {i}/364868892 records...')
#             if i> 0:
#                 break

train_data.to_json(os.path.join(save_path, dataset_output_name),  lines=True)


end_time = time.time()
print(f"Convert to json Elapsed time: {end_time - start_time:.4f} seconds")

hf_hub_download(repo_id="gpt2", filename="merges.txt", local_dir=save_path)
hf_hub_download(repo_id="gpt2", filename="vocab.json", local_dir=save_path)