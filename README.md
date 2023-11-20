This repository is a fork of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/). The original README can be found [here](Megatron.md).

# Zero Bubble Pipeline Parallelism

Zero Bubble Pipeline Parallelism is a novel pipeline parallelism algorithm able to reduce the bubble of pipeline parallelism to almost zero while preserving synchronous semantics.

Our paper is coming soon.

**Quick settings to enable Zero Bubble:**
```
  --zero-bubble-v-schedule
  --allow-padding-num-layers
  --enable-optimizer-post-validation
```
Can also try out with
`ZERO_BUBBLE_V_SCHEDULE=1 examples/pretrain_zero_bubble.sh`

**Acceleration**

Experiments shows zero bubble pipeline parallelism can accelerate training up to 30% with a similar memory comsumption. A detailed table of experiments is coming soon.

**Notices**
* ZBV schedule requires the number of layers per pipeline to be an even number, so that each stage can be splited into two virtual stages evenly.
* To achieve a better throughput, we recommend setting `--num-layers` to a value to `k * pipeline-model-parallel-size - 2` where k can be any value $\gt1$. This is used to compensate for the additional embedding layer on the first/last pipeline stages which could otherwise brings bubble to all other stages.

## Zero Bubble Schedules
The key of achieving zero bubble is to breaking a backward pass into a $B$ pass and $W$ pass. $B$ on one stage will only depend on the $B$ on its next stage, compared to depending on both $B$ and $W$ of in 1F1B.
![image](https://hackmd.io/_uploads/Bkc7CL7N6.png)

### Comparision of Schedules
* 1F1B
![image](https://hackmd.io/_uploads/Hkq-gD7N6.png)
* ZB1P
![image](https://hackmd.io/_uploads/Hy2GxwmEa.png)
* ZB2P
![image](https://hackmd.io/_uploads/S10QgvmV6.png)
* ZBV
![image](https://hackmd.io/_uploads/HkCgLKEET.png)


    

|                                                       | 1F1B    | ZB1P     | ZB2P | ZBV (Recommended) |
| ----------------------------------------------------- | ------- | -------- | ---- | --- |
| Bubble Rate                                           | $(p-1)/m$ | $(p-1)/3m$ | 0    | 0   |
| Activation Memory <br> (Compared to 1F1B)             | 1x       | 1x        | 2x    | 1x   |
| Pipeline Communication Volume <br> (Compared to 1F1B) | 1x       | 1x        | 1x    | 2x   |



<p style="font-size:14px;margin-bottom:0;height:20px;">* p: number of pipeline stages; m: number of microbatches</p>
<p style="font-size:14px;margin-bottom:0;height:20px;">* Assuming T<sub>F</sub> = T<sub>B</sub> = T<sub>W</sub></p>
<p style="font-size:14px;margin-bottom:0;height:20px;">* Communication volume of DP and TP stays the same</p>


## Zero Bubble Command Line Arguments

* `--enable-zero-bubble` Enables zero bubble schedules.
* `--zero-bubble-v-schedule` Enables ZBV schedule recommended above. Implies `--enable-zero-bubble`.
* `--enable-optimizer-post-validation` Enables optimizer post validation explained in [Optimizer Post Validation](#Optimizer-Post-Validation)
* `--allow-padding-num-layers` Allowing the number of layers to NOT be a mutiple of number of Pipelines. This allows us to have one less layer on the first and last pipeline stage to compensate for the bubble caused by embedding layers.
* `--zero-bubble-max-pending-backward` Controls memory limit of zero bubble schedules. Setting this to 1 x number of pipelines will get a schedule like ZB1P while setting to 2x number of pipelines will get ZB2P. No effect for ZBV schedule enabled by `--zero-bubble-v-schedule`.
* `--zero-bubble-pipeline-timers-start-iter` and `--zero-bubble-pipeline-timers-end-iter` Used to control the start/end iterations when ZB scheduler profiles each F/B/W to measure $T_F$, $T_B$ and $T_W$

## Optimizer Post Validation

In most practices of PP there's an all-reduce cross all pipeline stages for numerical robustness, e.g. global gradient norm for gradient clipping. INF/NAN check for mixed precision training, etc. This all-reduce breaks parallelogram and makes zero bubble impossible.
Under the observation that during a stable training both the gradient clipping and INF/NAN rarely triggers, we replace the before-hand synchronizations with a post update validation.

![image](https://hackmd.io/_uploads/B16R3q4N6.png)

We eagerly step the optimizers assuming the grad cliping, INF/NAN conditions are not triggered. In case an amendment to the gradient is required, a rollback will be issued and then we redo the optimizer step based on the fully reduced global state.

To enable this feature, add `--enable-optimizer-post-validation`. Experiments shows NOT enabling this will cause ~8% performance loss.