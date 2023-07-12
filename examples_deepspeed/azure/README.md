## Recipes for experimentation on Azure

The recipes have been tested on command line on a cluster setup using Azure VMs and VMSS as well as inside Docker based environments.

To run any of the examples in this folder, please go to the base directory of Megatron-DeepSpeed and run as follows

```bash examples_deepspeed/azure/run-benchmark-model.sh```

### Pre-requisites

To run the above script, you will need to either setup your own dataset and modify the scripts or use our helper scripts to download the publicly available Books dataset and GPT vocab files. Please use the following from the ```dataset``` folder

```bash dataset/download_books.sh```

```bash dataset/download_vocab.sh```

### Run 175B and 1T models

We have included two recipes for the 175B model and the 1T model. To train the model, we assume that the users will modify and tune hyperparameters and configurations by themselves. To facilitate initial training, we have made the recipes runnable with the Books dataset as follows.

```bash examples_deepspeed/azure/run-175b.sh```

```bash examples_deepspeed/azure/run-1t.sh```

### Note about ZeRO stage 3 and CPU offload 

By default, we have enabled ZeRO Stage 3 for both the recipes above. For the 1T model, we have also enabled the CPU-offload feature to save on memory and enable a larger batch size that offers better performance. 
