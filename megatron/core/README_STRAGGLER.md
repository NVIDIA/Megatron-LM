## StragglerDetector

The file `megatron/core/utils.py` has a class named `StragglerDetector` which supports Python Contexts
This class supports collecting timing events for various steps of a given iteration. It
keeps collecting such timing events on a per rank basis, and when the reporter is invoked
during a logging interval, it computes the min and max of certain metric across all
ranks and logs the observed metric and the rank as follows

```
 0: INFO:megatron.core.utils:[2024-03-14 23:07:56] | MnRtt/Rnk: 3453.08ms/8 | MxRtt/Rnk: 3468.20ms/0 | MnPwr/Rnk: 601796W/8 | MxPwr/Rnk: 683801W/18 | MnTmp/Rnk: 52C/0 | MxTmp/Rnk: 65C/21 | MnUtl/Rnk: 97%/8 | MxUtl/Rnk: 100%/6 | MnClk/Rnk: 1950MHz/28 | MxClk/Rnk: 1980MHz/0 | MnDRtt/Rnk: 14.27us/23 | MxDRtt/Rnk: 34.65us/3 | MnEtpt/Rnk: 296.02TF/0 | MxEtpt/Rnk: 297.32TF/8
```
<hr>

### Description of the metrics

Each metric is prefixed with `Mn` or `Mx` to represent `Minimum` or `Maximum`. Each metric is also suffixed with the rank where the metric was measured. The metrics are averaged over the logging interval. Between the prefix and the rank is the name of the metric as follows

- Rtt : RoundTrip Time (time spent in all the traced ops per iteration)
- Pwr : GPU Power
- Tmp : GPU Temperature
- Utl : GPU Utilization
- Clk : GPU Clock
- DRtt: get_batch latency
- Etpt: Estimated throughput. This is derived from actual computed throughput dividied by Rtt. Since we do not collect timing for backward pass, the value is further divided by three to come up with estimated throughput. 
<hr>

### Command Line activation
To start using the StragglerDetector, need to pass the following argument `--log-straggler`. It optionally also takes two additional parameters. Default disabled
- `--disable-straggler-on-startup` - whether to keept the StragglerDetector disabled on startup and enable later. Default enabled
- `--straggler-ctrlr-port` - The StragglerDetector can toggle between on/off just by sending `curl Rank0Host:port`. Default port is 65535. Every time it is turned 
- `--straggler-minmax-count` - If set to > 1 (N), it prints N Top and Bottom Etpt/Rank pairs as shown below
```
 0: INFO:megatron.core.utils:^^^^ Bottom 4 Ranks with lowest  Etpt(TF): 296.02/0, 296.17/2, 296.23/1, 296.23/4,
 0: INFO:megatron.core.utils:^^^^ Top    4 Ranks with highest Etpt(TF): 297.28/15, 297.28/11, 297.32/12, 297.32/8,
```
<hr>

### Programming the StragglerDetector
The StragglerDetector class supports context, and its implementation is a Singleton.
- Initialization 

```
 # initialization, where StragglerDetector will be used
   from megatron.core.utils import StragglerDetector
   stimer = StragglerDetector()
```

- One time for each rank

```
 # one time before the training loop starts
 stimer.configure(world, rank, enabled=True, port=65545)

 # Arguments to configure 
 #     world   : World Size
 #     rank    : The rank of this trainer
 #     mmcnt   : (Optional) Number of ranks to print for showing Min/Max Etpt
 #     amp     : (Optional) Set to 3.0 if we only use timers in fwd pass
 #     port    : (Optional) control port, useful only for rank-0
 #     prefill : (Optional) howmany Events to pre-populate
 #     enabled : (Optional) whether or not collection is enabled on startup
```

- To Capture time

```
 # whereever timing need to be captured
 with stimer:
     do_operation()

 # special case for get_batch
 with stimer(bdata=True):
      input,... = get_batch(iterator,...)
```

- Logging in main training loop

```
 # logging
   total_flops = 0.0
   iteration = 0
   # inside the main training loop
   while training:
        iteration += 1
        do_step()
        total_flops += get_computed_flops()
        if iteration % log_interval:
           stimer.report(total_flops, log_interval)
           total_flops = 0.0
```
