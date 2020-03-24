"""Race."""

from megatron.model.multiple_choice import MultipleChoice
from megatron.utils import print_rank_0
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune
from tasks.race.data import RaceDataset


def train_valid_datasets_provider(args):
    """Provide train and validation datasets."""

    train_dataset = RaceDataset('training', args.train_data,
                                args.tokenizer, args.seq_length)
    valid_dataset = RaceDataset('validation', args.valid_data,
                                args.tokenizer, args.seq_length)

    return train_dataset, valid_dataset


def model_provider(args):
    """Build the model."""

    print_rank_0('building multichoice model for RACE ...')

    return MultipleChoice(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        checkpoint_activations=args.checkpoint_activations)


def metrics_func_provider(args):
    """Privde metrics callback function."""

    def single_dataset_provider(datapath, args):
        name = datapath.split('RACE')[-1].strip('/').replace('/', '-')
        return RaceDataset(name, [datapath], args.tokenizer, args.seq_length)

    return accuracy_func_provider(args, single_dataset_provider)


def main(args):

    finetune(args, train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)
