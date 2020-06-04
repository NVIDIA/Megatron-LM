import lucene
import sys

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.util import Version

import torch
import torch.distributed as dist

from indexer import get_ict_dataset, get_one_epoch_dataloader
from megatron.initialize import initialize_megatron
from pretrain_bert_ict import get_batch


def setup():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])


def run(embed_all=False):
    dset = get_ict_dataset(use_titles=False, query_in_block_prob=0.1)
    dataloader = iter(get_one_epoch_dataloader(dset))

    index_dir = SimpleFSDirectory(Paths.get("full_wiki_index/"))
    analyzer = StandardAnalyzer()
    analyzer.setMaxTokenLength(1024)

    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(index_dir, config)

    # field for document ID
    t1 = FieldType()
    t1.setStored(True)
    t1.setTokenized(False)

    # field for document text
    t2 = FieldType()
    t2.setStored(True)
    t2.setTokenized(True)
    t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    correct = total = 0
    round_correct = torch.zeros(1).cuda()
    round_total = torch.zeros(1).cuda()
    for round in range(100000):
        with torch.no_grad():
            try:
                query_tokens, query_pad_mask, \
                block_tokens, block_pad_mask, block_index_data = get_batch(dataloader)
            except:
                break

        # query_tokens = query_tokens.detach().cpu().numpy()
        block_tokens = block_tokens.detach().cpu().numpy()

        # query_strs = [dset.decode_tokens(query_tokens[i].tolist(), hardcore=True) for i in range(query_tokens.shape[0])]
        block_strs = [dset.decode_tokens(block_tokens[i].tolist(), hardcore=True) for i in range(block_tokens.shape[0])]

        def add_document(text, writer, doc_id):
            doc = Document()
            doc.add(Field("text", text, t2))
            doc.add(Field("doc_id", doc_id, t1))
            writer.addDocument(doc)

        # add documents to index writer
        for i in range(len(block_strs)):
            add_document(block_strs[i], writer, i)

        # write and finalize the index
        writer.commit()

        # define BM25 searcher
        # searcher = IndexSearcher(DirectoryReader.open(index_dir))
        # searcher.setSimilarity(BM25Similarity())

        # # feed queries and get scores for everything in the index
        # hits_list = []
        # for s in query_strs:
        #     query = QueryParser("text", analyzer).parse(s)
        #     hits = searcher.search(query, 1).scoreDocs
        #     hits_list.append(hits)

        # for (i, hits) in enumerate(hits_list):
        #     doc_ids = [int(searcher.doc(hit.doc)['doc_id']) for hit in hits]
        #     correct += int(i in doc_ids)
        #     total += 1

        # dist.all_reduce(round_correct)
        # dist.all_reduce(round_total)

        # correct += int(round_correct.item())
        # total += int(round_total.item())

        # round_correct -= round_correct
        # round_total -= round_total

        # print("Correct: {:8d}   |   Total: {:8d}   |   Fraction: {:6.5f}".format(correct, total, correct / total))
        if round % 10 == 0:
            print(round)
    writer.close()

    # Plan
    # overall accuracy test:
    # have index with all blocks. For BERT these are token ids, for BM25 these are tokens
    #
    # 1. run batch size 4096 BM25 self similarity test. For this I can just detokenize out of the dataset.
    # I get the retrieval scores in the forward_step and log the results.
    # 2. Create a BM25 index over all of wikipedia, have it ready for use in megatron QA.
    #
    # Create an index with the block embeddings with block ids

if __name__ == "__main__":
    setup()
    run()
