import torch

from megatron import mpu, get_args
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.global_vars import set_global_variables, rebuild_tokenizer
from megatron.model import GPTModel
from tools.merge_mp_partitions import split_into_partitions


def resize_word_embedding_trick(origin_model: GPTModel, padded_voca_size: int):
    mpu.initialize.set_tensor_model_parallel_world_size(1)
    mpu.initialize.set_tensor_model_parallel_rank(0)

    embedding = origin_model.language_model.embedding
    old_embedding = embedding.word_embeddings
    new_embedding = mpu.VocabParallelEmbedding(
        padded_voca_size, embedding.hidden_size, init_method=embedding.init_method)
    old_num_tokens, hidden_size = old_embedding.weight.data.shape
    with torch.no_grad():
        new_embedding.weight.data[:old_num_tokens, :hidden_size] = old_embedding.weight.data
    embedding.word_embeddings = new_embedding


def get_model(model_type):
    if model_type == 'GPT':
        from pretrain_gpt import model_provider
    else:
        raise Exception('unrecognized model type: {}'.format(model_type))

    model = model_provider()
    model = model.half()

    return model


def get_mp_split_args(parser):
    """Provide extra arguments required for splitting."""
    group = parser.add_argument_group(title='mp split')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['GPT'],
                       help='Type of the mdoel.')
    group.add_argument('--target-tensor-model-parallel-size', type=int, default=2,
                       help='Degree of tensor model parallelism in output model.')

    return parser


def main():

    # Args
    set_global_variables(extra_args_provider=get_mp_split_args,
                         args_defaults = {'use_cpu_initialization': True,
                                          'micro_batch_size': 1,
                                          'no_load_optim': True,
                                          'no_load_rng': True,
                                          'no_save_optim': True,
                                          'no_save_rng': True,
                                          'save_interval': 1})
    args = get_args()

    if args.pipeline_model_parallel_size > 1:
        print("Checkpoints with pipeline model parallelism are not currently supported.")
        exit()

    model_type = args.model_type
    orig_tensor_model_parallel_size = args.tensor_model_parallel_size
    assert orig_tensor_model_parallel_size == 1, 'Expects that we split one file checkpoint'
    tokenizer = rebuild_tokenizer(args)

    print('\n splitting model into partitions ...')
    print(' > number of partitions: {}'.format(orig_tensor_model_parallel_size))
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of tokens ................ {} '.format(
        tokenizer.vocab_size))
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(
        args.num_attention_heads))
    print('    maximum position embeddings ..... {}'.format(
        args.max_position_embeddings))

    print('Splitting one file model...')

    # load one file model
    mpu.initialize.set_tensor_model_parallel_world_size(1)
    mpu.initialize.set_tensor_model_parallel_rank(0)
    mpu.initialize.set_pipeline_model_parallel_world_size(1)
    mpu.initialize.set_pipeline_model_parallel_rank(0)
    orig_model = get_model(model_type)
    iteration = load_checkpoint(orig_model, None, None)

    # create partition models
    partitions = []
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    rebuild_tokenizer(args)

    # resize word embedding
    resize_word_embedding_trick(orig_model, args.padded_vocab_size)

    mpu.initialize.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    for rank in range(args.model):
        mpu.initialize.set_tensor_model_parallel_rank(rank)
        model_ = get_model(model_type)
        partitions.append(model_)

    partitions_params_gen = [partition.named_parameters()
                             for partition in partitions]
    for orig_name, origin_param in orig_model.named_parameters():
        print(' > working on {} ...'.format(orig_name))
        print('     merged         type: {}, size: {}'.format(
            origin_param.dtype, list(origin_param.size())))

        partitions_param = []
        for rank, partition_params_gen in enumerate(partitions_params_gen):
            partition_name, partition_param = next(partition_params_gen)
            assert partition_name == orig_name
            partitions_param.append(partition_param)
            print('     partition {}    type: {}, size: {}'.format(
                rank, partition_param.dtype, list(partition_param.size())))

        # For the non-parallel parameters, simply copy the original model values.
        if not hasattr(origin_param, 'tensor_model_parallel'):
            print('     none-parallel parameter, simple copy from origin model')
            with torch.no_grad():
                for part_param in partitions_param:
                    part_param.data.copy_(origin_param.data)
        # For parallel parameters, merge the values
        else:
            dim = origin_param.partition_dim
            stride = origin_param.partition_stride
            print(f'     parallel parameter split with stride {stride} along '
                  f'dimention {dim}')
            result = split_into_partitions(origin_param, args.target_tensor_model_parallel_size, dim, stride)
            assert len(result) == len(partitions_param)
            with torch.no_grad():
                for i in range(len(result)):
                    partitions_param[i].data.copy_(result[i].data)

    for rank, part_model in enumerate(partitions):
        mpu.initialize.set_tensor_model_parallel_rank(rank)
        print(f"> saving rank {rank}'s model")
        save_checkpoint(iteration, part_model, None, None)
    print('done :-)')
