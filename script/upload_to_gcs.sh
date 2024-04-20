#!/bin/bash
# NFS の保存ディレクトリ名を指定したら、同じ名前のディレクトリを `gs://abeja-dev/` に作成し、
# その中に指定した iteration の checkpoints をアップロードする。
# iteration を指定しない場合、すべての iteration をアップロードする。

if [ $# -lt 1 ]; then
    echo "at least 1 arguments are required: local_path, target_iteration (optional)"
    echo "all ckpts will be uploaded if target_iteration is not specified."
    echo "Usage: $0 /mnt/nfs/checkpoints/test_run_name 10000"
    echo "  test_run_name dir will be created in gs://abeja-dev/ and 10000 iteration's ckpts will be uploaded."
    exit 1
fi

LOCAL_DIR=$1
RUN_NAME=$(basename $1)
BUCKET_NAME_AND_PATH="gs://abeja-dev"

if [ -n "$2" ]; then
    # 引数がある場合、それを `dir_引数` 形式で表示
    ITER_DIR_NAME=$(printf "iter_%07d" $2)
else
    ITER_DIR_NAME="*"
fi

LOCAL_DIR_ITER=$LOCAL_DIR/$ITER_DIR_NAME
GCS_PATH=$BUCKET_NAME_AND_PATH/$RUN_NAME

if ! command -v gsutil &> /dev/null; then
    echo "Error: 'gsutil' コマンドがインストールされていません。"
    exit 1
fi

echo "RUN: gsutil -m cp -r $LOCAL_DIR_ITER $GCS_PATH/"
gsutil -m cp -r $LOCAL_DIR_ITER $GCS_PATH/
