# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron import get_args, print_rank_0
from megatron.checkpointing import load_biencoder_checkpoint
from megatron.data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from megatron.data.realm_index import OpenRetreivalDataStore, FaissMIPSIndex
from megatron.model.biencoder_model import get_model_provider
from megatron.training import get_model
from tasks.orqa.unsupervised.nq import get_nq_dataset
from tasks.orqa.unsupervised.nq import get_one_epoch_nq_dataloader
from tasks.orqa.unsupervised.nq import process_nq_batch
from tasks.orqa.unsupervised.qa_utils import calculate_matches


class ORQAEvaluator(object):
    def __init__(self):
        args = get_args()
        self.embedding_size = args.hidden_size
        self.faiss_use_gpu = args.faiss_use_gpu
        self.evidence_embedder_obj = None
        self.evidence_dataset = None
        self.mips_index = None
        self.eval_dataset = None

        # Get Evidence (Wikipedia) dataset
        self.get_evidence_dataset()

        # Load query encoder checkpoint
        only_query_model = True
        if args.biencoder_shared_query_context_model:
            only_query_model = False

        model = get_model(get_model_provider(only_query_model=only_query_model,
            biencoder_shared_query_context_model=args.biencoder_shared_query_context_model))

        self.model = load_biencoder_checkpoint(model,
                only_query_model=only_query_model)

        assert len(self.model) == 1
        self.model[0].eval()

        # Load faiss indexer
        self.faiss_wrapper()

    def get_evidence_embedding(self):
        # This will load the embedding from the embedding path
        self.evidence_embedder_obj = OpenRetreivalDataStore(load_from_path=True)

    def get_evidence_dataset(self):
        self.evidence_dataset = get_open_retrieval_wiki_dataset()

    def faiss_wrapper(self):
        # Initialize FAISS wrapper on local rank = 0 as the evidence embeddings
        # is distributed over all the GPUs in a node and FAISS is not 
        # thread-safe
        args = get_args()
        if args.local_rank == 0:
            # Get evidence embeddings computed using context encoder
            self.get_evidence_embedding()

            assert self.evidence_embedder_obj is not None
            self.mips_index = FaissMIPSIndex(embed_size=self.embedding_size,
                                        embed_data=self.evidence_embedder_obj,
                                        use_gpu=self.faiss_use_gpu)

        # Wait for the FAISS index to be initialized in all the nodes
        torch.distributed.barrier()

    def generate_query_vectors(self, qa_data, split):

        self.eval_dataset = get_nq_dataset(qa_data, split)
        dataloader = get_one_epoch_nq_dataloader(self.eval_dataset)

        query_vectors = []
        reference_list = []

        for batch in dataloader:
            # batch also has query_tokens and query_pad_data
            query_tokens, query_mask, query_types, \
                query_len, reference = process_nq_batch(batch)

            assert len(self.model) == 1
            unwrapped_model = self.model[0]
            while not hasattr(unwrapped_model, 'embed_text'):
                unwrapped_model = unwrapped_model.module

            with torch.no_grad():
                query_logits = unwrapped_model.embed_text(
                    unwrapped_model.query_model, query_tokens, 
                    query_mask, query_types)

            reference_list.extend(reference)
            query_vectors.extend(query_logits.split(1, dim=0))
            if len(query_vectors) % 100 == 0:
                print_rank_0('Encoded queries {}'.format(len(query_vectors)))

        query_tensor = torch.cat(query_vectors, dim=0)
        print_rank_0('Total encoded queries tensor {}'.format(query_tensor.size()))

        assert query_tensor.size(0) == len(self.eval_dataset)
        return query_tensor, reference_list

    def evaluate(self, qa_data, split):
        args = get_args()
        query_tensor, reference_list = self.generate_query_vectors(qa_data, \
                                                                    split)
        local_rank = args.local_rank
        rank = torch.distributed.get_rank()
        device_count = torch.cuda.device_count()
        num_nodes = torch.distributed.get_world_size() // device_count
        node_id = rank // device_count

        for node in range(num_nodes):
            start_rank = node * device_count
            end_rank = (node + 1) * device_count
            ranks_list = list(range(start_rank, end_rank))
            node_group = torch.distributed.new_group(ranks=ranks_list)

            if node_id == node:
                device_start_rank = start_rank
                group = node_group
        
        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        tensor_list = [torch.empty_like(input_) for _ in range(device_count)]
        torch.distributed.all_gather(tensor_list, query_tensor, group=group)

        if local_rank == 0 and self.mips_index is not None:
            all_query_tensor = torch.cat(tensor_list, dim=0).contiguous()

            distance, topkindex = self.mips_index.search_mips_index(
                all_query_tensor, top_k=args.faiss_topk_retrievals, 
                reconstruct=False)
            distance = torch.from_numpy(distance).cuda()
            topkindex = torch.LongTensor(topkindex).cuda()

        if local_rank != 0:
            distance = torch.empty(device_count * len(query_tensor), \
                args.faiss_topk_retrievals, dtype=torch.float32).cuda()
            topkindex = torch.empty(device_count * len(query_tensor), \
                args.faiss_topk_retrievals, dtype=torch.int64).cuda()

        torch.distributed.broadcast(distance, src=device_start_rank, \
            group=group)
        torch.distributed.broadcast(topkindex, src=device_start_rank, \
            group=group)

        distance = torch.split(distance, len(query_tensor), dim=0)\
            [local_rank]
        topkindex = torch.split(topkindex, len(query_tensor), dim=0)\
            [local_rank]

        top_ids_and_scores = []
        for darray, topkarray in zip(distance, topkindex):
            top_ids_and_scores.append((topkarray.tolist(), darray.tolist()))

        passages = self.evidence_dataset.id2text
        match_stats = calculate_matches(passages,
                                        reference_list,
                                        top_ids_and_scores,
                                        workers_num=args.num_workers,
                                        match_type=args.faiss_match)
        top_k_hits = match_stats.top_k_hits

        print_rank_0("{} SET RESULTS".format(split))
        print_rank_0("topk-{} documents hits {}".format(
            args.faiss_topk_retrievals, top_k_hits))
        top_k_hits = [v / len(top_ids_and_scores) for v in top_k_hits]
        print_rank_0("top-k documents hits accuracy {}".format(top_k_hits))

        for i in args.retriever_report_topk_accuracies:
            print_rank_0("top-{}: {:.2f}".format(i, top_k_hits[i-1] * 100))

        return
