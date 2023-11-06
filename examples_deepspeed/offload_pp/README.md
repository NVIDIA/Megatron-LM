# ZeRO-Offload++ Tutorials

This folder contains examples that demonstrate how to use the new ZeRO-Offload++ features. 

ZeRO-Offload++ now supports **Twin-Flow** feature.

## Twin-Flow

Instead of all-or-nothing offloading strategy, **Twin-Flow** allows a portion of data to run on CPU and the other part on GPU simultaneously. Thus, we not only mitigate the memory pressure on GPU side by offloading data to CPU, but also utilize both CPU and GPU computation resources more efficiently. 

![Twin-Flow-img](./twin-offload.png)

As shown in above Figure, when ZeRO-Offload is triggered, **Twin-Flow** now allow user to set a new configuration arguement called `ratio` (default value == 1) to adjust the portion of parameter updates on CPU optimizer. For example, if this `ratio==0.4`, it means 0-40% of parameters are updated using CPUAdam on CPU side, while the rest 60% parameters are updatedusing FusedAdam on GPU side.

## How to use

Now **Twin-Flow** can be used at ZeRO stage 3 with Offload. Below we provide two tutorial examples on how to use **Twin-Flow**.

### DeepSpeed Toy Example

Here is a toy example for using **Twin-Flow** inside DeepSpeed repo. 

Under `/tests/small_model_debugging/` folder, Run 

```
deepspeed partial_offload_test.py --zero 3
```

### GPT Model Training in Megatron-DeepSpeed

To enable **Twin-Flow** here, we need to add two flags for Megatron configs as follows: 

#### Megatron Configurations
```
--no-pipeline-parallel \
--cpu-optimizer \
```
which have been added to `ds_pretrain_gpt_350M.sh`

#### DeepSpeed Configurations
On the DeepSpeed side, we need to add follow configurations:

```
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "ratio": 0.3
    }
```

Basically, we need to first enable CPU Offload. Then user can adjust the portion of parameter updating on CPU by adjusting `ratio` here. Its default value is 1, which means all parameter updates happen on CPU side. The above config example with ` "ratio" : 0.3` meaning 0-30% parameters are updating on CPU side, while the other 70% parameter updates happens on GPU side.

#### Tuning suggestion on ratio

To get best performance, we recommend to set this `ratio` value as low as possible without causing GPU memory Out-Ouf-Memory issue.

One additional config on DeepSpeed side is 

```
  "prescale_gradients": false,
```
mainly because right now ZeRO-3 does not support prescale gradients.

All above configs have been added to `ds_config_gpt_TEMPLATE.json`

#### End-to-end Training

To run a sample training of GPT-350M model using Megatron-Deepspeed, simply run as follows:

```
bash ds_pretrain_gpt_350M.sh
```

Now the training start running with **Twin-Flow**. Enjoy!

## On-going optimizations

We have some other features inside ZeRO-Offload++ which will come soon, stay tuned!

* Removing uncessary D2H memcpy in ZeRO-offload
* On-the-fly fp16 to fp32 data casting inside CPUAdam
