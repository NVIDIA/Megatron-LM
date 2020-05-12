from collections import defaultdict
import time
import pickle

import faiss
from faiss import index_factory, index_cpu_to_gpu
import numpy as np

from megatron import get_args


PCAS = [
    'PCA', 'PCAR', 'PCAW', 'PCAWR'
]

# PCA to 64 dim gets "first missing" ~ 95% and "mixed" ~ 5% for all
# however, this is pretty hard since the embeds and queries are totally random, would be better to test according to a distribution
# update: Using realisitc mean and covariance helps, but then adjusting for inner product makes it unusable again
# CONCLUSION: PCA should not be used for MIPS


QUANTIZERS = [
    'IVF4096_SQ16', # 'IMI2x9',
    'HNSW32_SQ16', # 'IVF4096_HNSW32'
]

# IMI2x9 or any other MultiIndex doesn't support inner product so it's unusable
# IVF4096_HNSW32 doesn't support inner product either



ENCODINGS = [
    'Flat',
    'PQ16np', # PQ16, PQ16x12(np)
    'SQ4', 'SQ8', 'SQ6', 'SQfp16',
  # 'LSH', 'LSHrt', 'LSHr', 'LSHt'
]

# PQ16 is pretty slow for creating and adding - ~96s for 1e5, 105s for 1e6
# PQ16np is a bit faster but is pretty inaccurate - misses top-1 result 2/3 of time (1e6 embeds)
# PQ16x12(np) gets real slow. Uses 4096 centroids.

# SQfp16 is solid.

# LSH is inaccurate - pretty much always missing the top-1 result (1e6 embeds)




def latest(times):
    return times[-1] - times[-2]


def get_embed_mean_and_cov():
    embed_data = pickle.load(open('/home/dcg-adlr-nkant-data.cosmos1202/hash_data/normed4096_whitened.pkl', 'rb'))
    embed_mean = embed_data['embed_mean']
    whitener = embed_data['embed_whitener']
    embed_cov = whitener.dot(whitener.transpose())

    return embed_mean, embed_cov


def get_embeds_and_queries(mean, cov, num_embeds, num_queries):
    embeds = np.random.multivariate_normal(mean, cov, num_embeds).astype('float32')
    queries = np.random.multivariate_normal(mean, cov, num_queries).astype('float32')
    return embeds, queries


def get_random_embeds_and_queries(d, num_embeds, num_queries):
    embeds = np.random.rand(num_embeds, d).astype('float32')
    queries = np.random.rand(num_queries, d).astype('float32')
    return embeds, queries



def print_timing_stats(name, create_and_add, search):
    print('{:20s} Create and add embeds: {:10.4f}s  |  Search embeds: {:10.4f}s'.format(name, create_and_add, search))


def print_accuracy_stats(name, gold_indices, estimated_indices):
    gold_indices, estimated_indices = list(gold_indices), list(estimated_indices)
    results = defaultdict(int)

    for gold, estimated in zip(gold_indices, estimated_indices):
        if gold[0] not in estimated:
            results['first_missing'] += 1
        elif np.array_equal(gold, estimated):
            results['all_equal'] += 1
        else:
            results['mixed'] += 1
    result_strs = ['first_missing', 'all_equal', 'mixed']
    print('{:20s} First missing: {:4d}  |  All equal: {:4d}  |  Mixed: {:4d}'.format(name, *[results[s] for s in result_strs]))


def create_and_test_gold(d, k, embeds, queries):
    times = [time.time()]
    res = faiss.StandardGpuResources()
    gold_idx = index_cpu_to_gpu(res, 0, index_factory(d, 'Flat'))
    gold_idx.add(embeds)
    times.append(time.time())
    create_and_add = latest(times)

    distances, indices = gold_idx.search(queries, k)
    times.append(time.time())
    print_timing_stats('Flat', create_and_add, latest(times))
    print('-' * 100)
    return distances, indices


def test_pca(d, k, embeds, queries, pca_dim):

    distances, indices = create_and_test_gold(d, k, embeds, queries)

    times = [time.time()]
    all_pca_indices = []
    for s in PCAS:
        pca_idx = index_factory(d, s + "{},Flat".format(pca_dim), faiss.METRIC_INNER_PRODUCT)
        pca_idx.train(embeds)
        pca_idx.add(embeds)
        times.append(time.time())
        create_and_add = latest(times)

        pca_distances, pca_indices = pca_idx.search(queries, k)
        all_pca_indices.append(pca_indices)
        times.append(time.time())
        print_timing_stats(s, create_and_add, latest(times))

    print('\n')
    for s, pca_indices in zip(PCAS, all_pca_indices):
        print_accuracy_stats(s, indices, pca_indices)


def test_quantizers(d, k, embeds, queries):

    distances, indices = create_and_test_gold(d, k, embeds, queries)

    times = [time.time()]
    for s in QUANTIZERS:
        if 'HNSW' in s:
            quant_idx = index_factory(d, s, faiss.METRIC_INNER_PRODUCT)
        else:
            quant_idx = index_factory(d, "Flat," + s, faiss.METRIC_INNER_PRODUCT)

        quant_idx.train(embeds)
        quant_idx.add(embeds)
        times.append(time.time())
        create_and_add = latest(times)

        quant_distances, quant_indices = quant_idx.search(queries, k)
        times.append(time.time())
        print_timing_stats(s, create_and_add, latest(times))


def test_encodings(d, k, embeds, queries):

    distances, indices = create_and_test_gold(d, k, embeds, queries)

    times = [time.time()]
    all_encode_indices = []
    for s in ENCODINGS:
        encode_idx = index_factory(d, s, faiss.METRIC_INNER_PRODUCT)

        encode_idx.train(embeds)
        encode_idx.add(embeds)
        times.append(time.time())
        create_and_add = latest(times)

        _, encode_indices = encode_idx.search(queries, k)
        all_encode_indices.append(encode_indices)
        times.append(time.time())
        print_timing_stats(s, create_and_add, latest(times))

    print('\n')
    for s, encode_indices in zip(ENCODINGS, all_encode_indices):
        print_accuracy_stats(s, indices, encode_indices)


def run_all_tests():
    mean, cov = get_embed_mean_and_cov()
    embeds, queries = get_embeds_and_queries(mean, cov, int(1e6), 256)
    d = 128
    k = 10
    test_pca(d, k, embeds, queries, 96)
    test_quantizers(d, k, embeds, queries)
    test_encodings(d, k, embeds, queries)




if __name__ == "__main__":
    run_all_tests()





