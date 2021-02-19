# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import json
from lsh import cache, minhash
import time
import pickle
import sys


# This function is adapted from:
#   https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram]
               for head in range(0, len(text) - char_ngram))


# This function is adapted from:
#  https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


if __name__ == '__main__':

    print('parsing the inputs ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs = '*', default=None, help = 'List of '
                        'the input files')
    parser.add_argument('--load-fingerprints', type=str, default=None,
                       help='Load the fingerprints from pickle file.')
    parser.add_argument('--save-fingerprints', type=str, default=None,
                       help='Save the fingerprints of the inputs.')
    parser.add_argument('--output', type=str,
                       help='Output file name.')
    args = parser.parse_args()

    print('finding possible duplicate content ...')

    hasher = minhash.MinHasher(seeds=100, char_ngram=5, hashbytes=4)
    lshcache = cache.Cache(bands=10, hasher=hasher)

    url_doc = {}

    # load fingerprints from pickle file if needed
    if args.load_fingerprints is not None:
        print("Loading fingerprints from pickle file {}".format(
            args.load_fingerprints), flush=True)
        with open(args.load_fingerprints, "rb") as f:
            lshcache = pickle.load(f)
            url_doc = pickle.load(f)

    counter = 0
    start_time = time.time()

    print("Computing fingerprints", flush=True)

    input_pairs = 0 if args.inputs is None else int(len(args.inputs)/2)
    for i in range(input_pairs):
        input_file = args.inputs[2 * i]
        key = args.inputs[2 * i + 1]
        print(' document processing {} with key {}'.format(input_file, key),
            flush=True)
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    myjson = json.loads(line)
                    url = myjson[key]
                    text = myjson['text']
                    counter += 1
                    url_doc[url] = text
                    lshcache.add_fingerprint(hasher.fingerprint(text), url)
                except Exception as e:
                    print('Error:', e)
                if counter % 10000 == 0:
                    print(' [read]> processed {} documents in {:.2f} '
                        'seconds ...'.format(counter, time.time() - \
                        start_time), flush=True)

    # Save the fingerprints if needed
    if args.save_fingerprints is not None:
        print("Saving fingerprints to pickle file {}".format(
            args.save_fingerprints), flush=True)
        with open(args.save_fingerprints, 'wb') as f:
            pickle.dump(lshcache, f)
            pickle.dump(url_doc, f)

    counter = 0
    start_time = time.time()
    deduped = 0
    with open(args.output, 'wb') as f:
        for b in lshcache.bins:
            for bucket_id in b:
                if len(b[bucket_id]) > 1:
                    items = list(b[bucket_id])
                    main_url = items[0]
                    main_dhingles = shingles(url_doc[main_url])
                    remove_urls = []
                    for i in range(1, len(items)):
                        counter += 1
                        other_url= items[i]
                        other_shingles = shingles(url_doc[other_url])
                        try:
                            jaccard_sim = jaccard(main_dhingles, other_shingles)
                        except Exception as e:
                            print('Error:', e)
                        if jaccard_sim > 0.5:
                            remove_urls.append({other_url: jaccard_sim})
                            deduped += 1
                        if counter % 10000 == 0:
                            print(' [write]> processed {} documents in {:.2f} '
                                  'seoncds and deduped {} documents ...'.
                                  format(counter, time.time() - start_time,
                                         deduped), flush=True)
                    if len(remove_urls) > 0:
                        myjson = json.dumps({main_url: remove_urls},
                                            ensure_ascii=False)
                        f.write(myjson.encode('utf-8'))
                        f.write('\n'.encode('utf-8'))

    print('done :-)')
