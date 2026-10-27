import glob
import json
import os
import tarfile

import webdataset as wds
from huggingface_hub import snapshot_download
from tqdm import tqdm


def _extract_archives(root: str):
    """Extract every .tar / .tar.gz archive found under *root* into its directory."""
    archives = glob.glob(os.path.join(root, "**", "*.tar*"), recursive=True)
    for arch in archives:
        try:
            print(f"Extracting {arch} …")
            with tarfile.open(arch, "r:*") as tf:
                tf.extractall(path=os.path.dirname(arch))
        except Exception as e:
            print(f"[WARN] Failed to extract {arch}: {e}")


def convert_llava_video_to_wds(dataset_root: str, shard_size: int = 8000):
    """Convert a LLaVA-Video dataset (keys: video, conversations, data_source) to WebDataset format.

    The function walks through every *.json / *.jsonl annotation file located under *dataset_root*,
    finds the referenced video files, and writes shards (<dataset_root>/wds/video-000000.tar …).
    """
    # ensure archives extracted so that video files are accessible
    _extract_archives(dataset_root)

    output_dir = os.path.join(dataset_root, "wds")
    os.makedirs(output_dir, exist_ok=True)

    # gather annotation files (skip the output directory itself)
    annotation_files = [
        p
        for p in glob.glob(os.path.join(dataset_root, "**", "*.json*"), recursive=True)
        if not os.path.commonpath([p, output_dir]) == output_dir
    ]
    if not annotation_files:
        raise FileNotFoundError(f"No annotation JSON files found in {dataset_root}")
    
    print(f"Found annotation files -  {annotation_files}")

    shard_pattern = os.path.join(output_dir, "video-%06d.tar")
    sample_idx = 0
    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
        for ann_path in annotation_files:
            print(f"Processing {ann_path} …")
            with open(ann_path, "r") as f:
                first = f.read(1)
                f.seek(0)
                entries = json.load(f) if first == "[" else [json.loads(line) for line in f if line.strip()]
            for entry in tqdm(entries):
                video_rel = entry.get("video")
                conversations = entry.get("conversations")
                if video_rel is None or conversations is None:
                    continue

                video_path = video_rel if os.path.isabs(video_rel) else os.path.join(dataset_root, video_rel)

                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    # or raise an error
                    continue

                try:
                    with open(video_path, "rb") as vf:
                        video_bytes = vf.read()
                except Exception:
                    continue

                key = f"{sample_idx:09d}"
                ext = os.path.splitext(video_path)[1].lstrip(".").lower() or "mp4"
                sample = {
                    "__key__": key,
                    ext: video_bytes,
                    "json": json.dumps(conversations).encode(),
                }
                if entry.get("data_source"):
                    sample["src.txt"] = str(entry["data_source"]).encode()

                sink.write(sample)
                sample_idx += 1

    print(f"Finished writing {sample_idx} samples → {output_dir}")


if __name__ == "__main__":
    # download dataset
    dataset_name = "lmms-lab/LLaVA-Video-178K"

    # specific subset to download
    subset = "0_30_s_academic_v0_1"

    dataset_root = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[f"{subset}/*", f"{subset}.*"],
    )
    print(f"dataset downloaded to: {dataset_root}")
    # convert to webdataset
    convert_llava_video_to_wds(f"{dataset_root}/{subset}")
