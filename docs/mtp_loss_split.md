
支持loss切分到不同rank

配置参数：
  pipeline_model_parallel_size: 8
  pipeline_model_parallel_layout: "E|(t|)*2m|m|m|LL|LL"
  num_layers: 2
  mtp_num_layers: 3

实现逻辑：
rank0上只有embedding，rank1和rank2上只有text model，rank3，rank4，rank5上只有mtp layer，总共三层mtp layer，分别是mtp layer0，mtp layer1，mtp layer2。rank6和rank7上只有output layer。
rank6和rank7各自计算两个 logist和loss，计算发生的顺序是mtp layer3，mtp layer2，mtp layer0，main model loss。所以rank6负责 mtp layer3，mtp layer2。rank7负责mtp layer0，main model loss.

先写设计文档，
有哪些需要选择的地方，先讲出来。
我来review，然后再写代码。
