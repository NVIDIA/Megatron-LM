import numpy as np
import torch
import torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.bert_dataset import get_indexed_dataset_
from megatron.data.ict_dataset import InverseClozeDataset
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from pretrain_bert_ict import model_provider


def main():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_checkpoint()
    dataset = get_dataset()

    num_docs = 100
    all_doc_logits = np.zeros(num_docs, 128)
    for i in range(num_docs):
        doc_tokens = []
        doc_token_lists = dataset.get_sentence_split_doc(i)
        ptr = 0
        while len(doc_tokens) < args.seq_length and ptr < len(doc_token_lists):
            doc_tokens.extend(doc_token_lists[ptr])

        doc_tokens, doc_token_types, doc_pad_mask = dataset.concat_and_pad_tokens(doc_tokens)
        doc_logits = model.embed_doc(np.array(doc_tokens), np.array(doc_pad_mask), np.array(doc_token_types))
        all_doc_logits[i] = doc_logits

    print(all_doc_logits, flush=True)


def load_checkpoint():
    args = get_args()
    model = get_model(model_provider)

    if isinstance(model, torchDDP):
        model = model.module
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    assert iteration > 0
    checkpoint_name = get_checkpoint_name(args.load, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def load_doc_embeds(path):
    pass


def get_dataset():
    args = get_args()
    indexed_dataset = get_indexed_dataset_(args.data_path, 'mmap', True)

    doc_idx_ptr = indexed_dataset.get_doc_idx()
    total_num_documents = indexed_dataset.doc_idx.shape[0] - 1
    indexed_dataset.set_doc_idx(doc_idx_ptr[0:total_num_documents])
    kwargs = dict(
        name='full',
        indexed_dataset=indexed_dataset,
        data_prefix=args.data_path,
        num_epochs=None,
        max_num_samples=total_num_documents,
        max_seq_length=288,  # doesn't matter
        short_seq_prob=0.01,  # doesn't matter
        seed=1
    )
    dataset = InverseClozeDataset(**kwargs)
    return dataset


if __name__ == "__main__":
    main()
