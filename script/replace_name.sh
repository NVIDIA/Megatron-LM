#!/usr/bin/env bash
# 用法:  bash replace_name.sh  input_A  input
# 功能:  把当前目录下“文件名后缀为 input_A”的所有文件/文件夹（递归）
#        改名为把最后的 input_A 替换成 input，保留其余部分及扩展名。

set -euo pipefail

[[ $# -eq 2 ]] || { echo "Usage: $0 <old_suffix> <new_suffix>"; exit 1; }

old_suffix=$1
new_suffix=$2

# 递归查找：类型 f 或 d，名字以 old_suffix 结尾
while IFS= read -r -d '' path; do
    dir=${path%/*}          # 目录部分
    name=${path##*/}        # 纯文件名
    new_name=${name%"$old_suffix"}${new_suffix}
    new_path="${dir}/${new_name}"

    # 若目标已存在则跳过
    [[ -e $new_path ]] && { echo "Skip (exists): $new_path"; continue; }

    echo "MV: $path  ->  $new_path"
    mv -- "$path" "$new_path"
done < <(find . -depth \( -type f -o -type d \) -name "*${old_suffix}" -print0)