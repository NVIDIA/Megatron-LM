import os
from datasets import load_dataset,concatenate_datasets
from tqdm import tqdm
dataset_root = "dataset/wikipedia"

configs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

all_datasets = []
for config in tqdm(configs):
    try:
        ds = load_dataset(dataset_root, config, split="train", keep_in_memory=False)
        all_datasets.append(ds)
    except Exception as e:
        print(f"Failed to load dataset for config {config}: {e}")

if all_datasets:
    # merged_dataset = all_datasets[0]
    merged_dataset = concatenate_datasets(all_datasets)
    # for ds in tqdm(all_datasets[1:]):
        # merged_dataset = merged_dataset.concatenate(ds)
    print("All datasets merged successfully.")
else:
    print("No datasets were loaded. Please check the dataset root directory and configurations.")

output_file = "dataset/wikipedia.json"
merged_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)

print(f"Merged dataset saved to {output_file}")
