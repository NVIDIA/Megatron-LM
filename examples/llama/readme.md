
Llama2/Llama3 Model pretraining instruction 

1. Environment setup 
   
   download docker image: xxxxx
   launch docker container: xxxx

2. Configurations in script (Megatron/examples/llama)
   
   -- network interface: change "ens50f0np0" to your system network interface, by running "ip a"  
      export NCCL_SOCKET_IFNAME=ens50f0np0  
      export GLOO_SOCKET_IFNAME=ens50f0np0 

   -- dataset: you can use both mock data and real data
      mock data: replace --data-path $DATA_PATH \ by --mock-data \
      real data: change the data path accordingly 
           DATA_DIR="/root/.cache/data"  # change to where the dataset is stored
           DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence

   -- Tokenizer: HuggingFaceTokenizer, Llama2Tokenizer
      
      for Llama2 training, we use Llama2Tokenizer
      
      for Llama3 training, we use HuggingFaceTokenizer, set huggingface model link in TOKENIZER_MODEL as below
      
      TOKENIZER_MODEL=meta-llama/Llama-3.1-8B #llama3      

   -- multi-node training: 
        MASTER_ADDR="${MASTER_ADDR:-localhost}" : change localhost to master node name
        NNODES="${NNODES:-1}" : change to # of nodes you want to train on, 2, 4, 8, etc. 
        NODE_RANK="${NODE_RANK:-0}" : change to the rank number of each node, 0, 1, 2, .. NNODES-1


3. How to run 

   --single node training: 
   TEE_OUTPUT=1 MBS=5 BS=120 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1 SEQ_LENGTH=4096 bash train_llama2.sh
   Sample output:
   ![alt text](image.png)


   --multi node training: 
   Launch the same docker container on each node (2, 4, etc.)
   run the training script on each node inside the container, start from master node, then slave node
   master: TEE_OUTPUT=1 MBS=4 BS=64 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1 SEQ_LENGTH=4096 bash train_llama2.sh
   slave:  TEE_OUTPUT=1 MBS=4 BS=64 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1 SEQ_LENGTH=4096 bash train_llama2.sh 
   Sample output (2-node):
      master node: 
      ![alt text](image-1.png)
      slave node:
      ![alt text](image-3.png)
   

4. Pay attention to key variables  
   -- TE_FP8: 0 - BP16, 1: FP8
   -- GEMM_TUNING: 1 - enable gemm tuning, which will boost performance by leveraging best gemm kernels
   -- USE_FLASH_ATTN: 1 to enable flash attention
   -- ENABLE_PROFILING : 1 to enable pytorch profiling for performance analysis 
   -- transformer-impl=transformer_engine : using transformer engine(TE), can set to local if you want to disable TE
   -- MODEL_SIZE: 7B, 70B for llama2, 8B, 70B for llama3/3.1 
   -- TOTAL_ITERS: 10 - total # of iterations 


   

    
      
   

