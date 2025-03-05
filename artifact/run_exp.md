## Run Container
```bash
docker run --name vocab_torch24 \
    --network=host -d -v "$(pwd):/vocab" \
    --runtime=nvidia --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --privileged=true \
    nvcr.io/nvidia/pytorch:24.05-py3 sleep infinity
```