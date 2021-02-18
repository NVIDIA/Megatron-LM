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

"""
Deduplicate downstream tasks from training dataset. 13-grams have been used.
All split documents with less than 200 characters got filtered. Any document
with more than 10 splits got filtered as well.
"""

from functools import partial
import json
import multiprocessing
import nltk
import re
import string
import sys
import time

def get_words(text):
    # get all the lowercase words from text
    words, positions = [], []
    for match in re.finditer(r'\w+', text.lower()):
        words.append(match.group(0))
        positions.append(match.start())
    return words, positions

def free_ngram(line, ngrams, ngram_size, filter_text_len, 
    splits_count, split_window_each_size):
    # remove all the ngrams

    try:
        myjson = json.loads(line)
        text_buf = [myjson['text']]
    except Exception as e:
        print("Error: {}".format(e), flush=True)
        text_buf = []

    text_buf_ngram_free = []
    while len(text_buf) > 0:

        # get the first one from the buffer
        text = text_buf.pop(0)
        words, positions = get_words(text)
        
        not_ngram_free = True
        punctuations = ".!?"
        # find n-grams
        for i in range(len(words) - ngram_size + 1):
            seq = " ".join(words[i:i+ngram_size])
            if seq in ngrams:

                # splits the text
                # first part of the text
                pos = positions[i] - split_window_each_size
                text_first = ""
                while pos > 0 and not text[pos] in punctuations:
                    pos -= 1
                if pos > 0:
                    text_first = text[0:pos+1]
                pos = positions[i] + split_window_each_size
                # last part of the text
                text_second = ""
                while pos < len(text) and not text[pos] in punctuations:
                    pos += 1
                if pos + 1 < len(text):
                    text_second = text[pos+1:len(text)]
                
                # first part of ngrams free
                if len(text_first) > filter_text_len:
                    text_buf_ngram_free.append(text_first)

                # add second part for further processing
                if len(text_second) > filter_text_len:
                    text_buf.append(text_second)
                not_ngram_free = False
                break

        # text are ngram free
        if not_ngram_free:
            text_buf_ngram_free.append(text)

    return text_buf_ngram_free


if __name__ == '__main__':

    print('finding possible duplicate content ...')
    main_file = sys.argv[1] # lambada file
    dedup_file = sys.argv[2] # Book corpus
    output_file = sys.argv[3] #Filtered book corpus
    ngrams = {}
    id_prefix = "lambada"

    # we use 13-grams, any text less than 200 characters got removed
    # any text splitted more than 10 got removed as well
    ngram_size = 13
    filter_text_len = 200
    splits_count = 10
    split_window_each_size = 200

    print('Reading file {} and computing ngrams'.format(main_file))
    with open(main_file, 'r') as f:
        for line in f:
            try:
                myjson = json.loads(line)
                words, positions = get_words(myjson['text'])
                for i in range(len(words) - ngram_size+1):
                    seq = " ".join(words[i:i+ngram_size])
                    if seq not in ngrams:
                        ngrams[seq] = positions[i]
            except Exception as e:
                print('Error:', e)
    print("ngrams size {}".format(len(ngrams)))

    print('Reading file {} and deduping n-grams'.format(dedup_file))
    counter = 0
    start_time = time.time()
    out_f = open(output_file, 'wb')
    splitted, ignored, split_mt_thld = 0, 0, 0

    # Setup multi-processing.
    num_workers = 40
    fin = open(dedup_file, 'r', encoding='utf-8')
    pool = multiprocessing.Pool(num_workers)
    free_ngram_x=partial(free_ngram, ngrams=ngrams, ngram_size=ngram_size, 
        filter_text_len=filter_text_len, splits_count=splits_count,
        split_window_each_size=split_window_each_size)
    free_ngrams = pool.imap(free_ngram_x, fin, 25)

    for text_buf_ngram_free in free_ngrams:
        counter += 1
        try:
            
            if len(text_buf_ngram_free) > 1:
                splitted += (len(text_buf_ngram_free) - 1)
            if len(text_buf_ngram_free) == 0:
                ignored += 1
            # more than 10 splits ignored
            if len(text_buf_ngram_free) > splits_count:
                text_buf_ngram_free = []
                split_mt_thld += 1

            for i in range(len(text_buf_ngram_free)):
                split_id_string = id_prefix + '-{:010d}'.format(int(counter)) \
                    + '-{:010d}'.format(int(i))
                outjson = json.dumps({"text":text_buf_ngram_free[i], 
                    id_prefix+"_split_id":split_id_string},
                    ensure_ascii=False)
                out_f.write(outjson.encode('utf-8'))
                out_f.write('\n'.encode('utf-8'))

            if counter % 1000 == 0:
                print(' [search]> processed {} documents in {:.2f} seconds ...'.
                    format(counter, time.time() - start_time), flush=True)
        except Exception as e:
            print('Error:', e)

    print("Deduped file written to: {}".format(output_file), flush=True)
    print("Total docs {} splitted {} ignored {} docs with many splits {}".\
        format(counter, splitted, ignored, split_mt_thld), flush=True)
    print('done :-)')
