1. docker image:  
Hopper: nvcr.io/nvidia/pytorch:25.04-py3  
Blackwell: ToDo

2. Clone MCore and switch to bagel branch  
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/zhuoyaow/megatron-lm-bagel.git  
cd megatron-lm-bagel
git checkout -b bagel origin/zhuoyaow/bagel_diffusion
git submodule update --init --recursive 
```

3. Install dependencies  
```bash
bash install_bagel.sh 
```

4. Download datasets   
```bash
wget -O bagel_example.zip \
  https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
unzip bagel_example.zip -d ./data
```

5. Run train  
```bash
bash examples/mimo/example_bagel_gen_training.sh
```



