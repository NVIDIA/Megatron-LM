import os
import glob
import subprocess

file_PATH = os.environ['RAW_DATA_FOLDER']+"/allam_data_2-1_splits/*"
print(file_PATH)
_dict = {}
for _file in glob.glob(file_PATH):
    if _file.endswith("wc.jsonl"):
        newfile_name = "_".join(_file.split("_")[:-2])+".jsonl"
        assert newfile_name not in _dict
        _dict[newfile_name] = 1
        cmd = f"mv {_file} {newfile_name}"
        print(cmd)
        subprocess.check_output(cmd, shell=True)