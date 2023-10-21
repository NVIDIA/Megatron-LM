# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""
Deduplicate downstream tasks from training dataset. 13-grams have been used.
All split documents with less than 200 characters got filtered. Any document
with more than 10 splits got filtered as well.
"""

import argparse
from functools import partial
import json
import multiprocessing
import nltk
import pickle
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

# splits the text
def split_text(text, start_position, remove_char_each_side, seq):
    # first part of the text
    punctuations = ".!?"
    pos = start_position - remove_char_each_side
    text_first = ""
    while pos > 0 and not text[pos] in punctuations:
        pos -= 1
    if pos > 0:
        text_first = text[0:pos+1]

    # add length of seq and remove_char_each_side
    pos = start_position + len(seq) + remove_char_each_side

    # last part of the text
    text_second = ""
    while pos < len(text) and not text[pos] in punctuations:
        pos += 1
    if pos + 1 < len(text):
        text_second = text[pos+1:len(text)]

    return text_first, text_second

def check_and_clean_text(args, words, ngrams, text, start_position, \
    text_buf_ngram_free, text_buf, local_ngram):

    seq = " ".join(words)
    if seq in ngrams:
        print(" [matched]: {}".format(seq), flush=True)

        if args.get_ngram_freq_only:
            # increase freq of this seq and then only consider the later part
            # of the text for further processing
            if seq in local_ngram:
                local_ngram[seq] += 1
            else:
                local_ngram[seq] = 1
            #print(" [increased]: {} {}".format(seq, ngrams[seq]), flush=True)
            if (start_position + len(seq) + 1) < len(text):
                text_buf.append(text[start_position + len(seq) + 1:len(text)])
            return False            

        # split the text
        text_first, text_second = split_text(text, start_position, \
            args.remove_char_each_side, seq)

        # first part of ngrams free
        if len(text_first) > args.filter_text_char_len:
            text_buf_ngram_free.append(text_first)

        # add second part for further processing
        if len(text_second) > args.filter_text_char_len:
            text_buf.append(text_second)

        return False # not ngram free

    # ngram free
    return True


def free_ngram(line, args, key, ngrams, ngrams_freq_sorted):
    # remove all the ngrams

    try:
        myjson = json.loads(line)
        text_buf = [myjson[key]]
    except Exception as e:
        print("Error: {}".format(e), flush=True)
        text_buf = []

    text_buf_ngram_free = []
    local_ngram = {}
    while len(text_buf) > 0:

        # get the first one from the buffer
        text = text_buf.pop(0)
        words, positions = get_words(text)
        
        ngram_free = True
        # find each max n-grams and check dictionary
        for i in range(len(words) - args.max_ngram_size + 1):
            check_ngram_free = check_and_clean_text(args, words[i:\
                i+args.max_ngram_size], ngrams, text, positions[i], \
                text_buf_ngram_free, text_buf, local_ngram)

            # the seq is ngram free? if yes, break
            if not check_ngram_free:
                ngram_free = False
                break

            # if max ngrams doesn't match, check if any other lower n-grams
            # within max ngram macthes
            for ngram_len, _ in ngrams_freq_sorted:
                check_ngram_free = check_and_clean_text(args, words[i:\
                    i+ngram_len], ngrams, text, positions[i], \
                    text_buf_ngram_free, text_buf, local_ngram)

                # same check as above
                if not check_ngram_free:
                    ngram_free = False
                    break

            # check break from lower than max ngram loop above
            if not ngram_free:
                break

        # for the last max n-gram, check all the lower ngrams in it
        if ngram_free and len(words) - args.max_ngram_size > 0:
            # get the last words of the lax max ngram
            last_seq_words = words[(len(words)-args.max_ngram_size):len(words)]
            last_seq_start_position = len(words) - args.max_ngram_size

            # check all n-grams lower than the max
            for pos, (ngram_len, _) in enumerate(ngrams_freq_sorted):

                # ignore the max ngram as has been considered already
                if ngram_len == args.max_ngram_size:
                    continue

                # find each ngram of ngram_len in max n-grams and check
                for i in range(len(last_seq_words) - ngram_len + 1):
                    check_ngram_free = check_and_clean_text(args, \
                        last_seq_words[i:i+ngram_len], ngrams, text,\
                        positions[last_seq_start_position+i], \
                        text_buf_ngram_free, text_buf, local_ngram)

                    if not check_ngram_free:
                        ngram_free = False
                        break

                if not ngram_free:
                    break

        # texts are ngram free
        if ngram_free and not args.get_ngram_freq_only:
            text_buf_ngram_free.append(text)

    # check if the text has only been trimmed
    trimmed = 0
    if not args.get_ngram_freq_only and len(text_buf_ngram_free) == 1 and \
        len(text_buf_ngram_free[0]) < len(myjson[key]):
        trimmed = 1

    return text_buf_ngram_free, trimmed, myjson, local_ngram

# insert word sequence into dictionary
def insert_dict(words, ngrams, pos):
    seq = " ".join(words)
    if seq not in ngrams:
        ngrams[seq] = 0
        #ngrams[seq] = pos

# insert each ngram from text into the ngrams dictionary
def compute_ngrams_insert_dict(args, text, ngrams):
    words, positions = get_words(text)
    if len(words) < args.min_ngram_size:
        return

    if len(words) < args.max_ngram_size:
        insert_dict(words, ngrams, positions[0])

    for i in range(len(words) - args.max_ngram_size+1):
        insert_dict(words[i:i+args.max_ngram_size], ngrams, positions[i])


# Build ngrams for the lambada dataset
def process_task_lambda(args, task_file, ngrams):
    print(' reading from {} and computing ngrams'.format(task_file))
    with open(task_file, 'r') as f:
        for line in f:
            try:
                myjson = json.loads(line)
                text = myjson['text']
                compute_ngrams_insert_dict(args, text, ngrams)
            except Exception as e:
                print('Error:', e)
    print(" Entities in ngrams {}".format(len(ngrams)), flush=True)


# Build ngrams for the dataset of the given task
def process_task(args, task_name, ngrams):

    print(' reading from {} and computing ngrams'.format('import datasets'))
    print(" Current entities in ngrams {}".format(len(ngrams)), flush=True)
    # using validation/test data from datasets
    from datasets import load_dataset

    entities_in_ngrams = len(ngrams)

    # load the dataset
    if task_name == 'squad':
        dataset = load_dataset('squad_v2', split='validation')
    elif task_name == 'natural_questions':
        dataset = load_dataset('natural_questions', split='validation')
    elif task_name == 'triviaqa':
        dataset = load_dataset('trivia_qa', 'unfiltered', split='test')
    elif task_name == 'webqa':
        dataset = load_dataset('web_questions', split='test')
    elif task_name == 'race':
        dataset = load_dataset('race', 'all', split='test')
    elif task_name == 'drop':
        dataset = load_dataset('drop', split='validation')
    elif task_name == 'coqa':
        dataset = load_dataset('coqa', split='validation')
    elif task_name == 'piqa':
        dataset = load_dataset('piqa', split='test')
    else:
        print("Invalid task name: {}".format(task_name), flush=True)
        return

    # read the dataset and add to ngrams
    for line in dataset:
        try:
            if task_name in ['squad', 'triviaqa', 'webqa', 'race', 'drop']:
                text = line['question']
                compute_ngrams_insert_dict(args, text, ngrams)
            elif task_name == 'natural_questions':
                text = line['question']['text']
                compute_ngrams_insert_dict(args, text, ngrams)
            elif task_name == 'coqa':
                all_questions = line['questions']
                for question in all_questions:
                    compute_ngrams_insert_dict(args, question, ngrams)
            elif task_name == 'piqa':
                text = line['goal']
                compute_ngrams_insert_dict(args, text, ngrams)
        except Exception as e:
            print('Error:', e)

    print(" After task {} entities in ngrams {}, added {}".format(task_name, \
            len(ngrams), len(ngrams) - entities_in_ngrams), flush=True)

def compute_tasks_ngrams(args, ngrams):
    start_time = time.time()
    for _, task_name in enumerate(args.tasks):
        print('Task: {}'.format(task_name), flush=True)
        if task_name == 'lambada':
            assert args.lambada_path is not None
            process_task_lambda(args, args.lambada_path, ngrams)
        else:
            process_task(args, task_name, ngrams)
    print(" Taken time to compute ngrams {:.2f}".format(time.time() - \
        start_time), flush=True)

def compute_ngram_freq_sorted(args, ngrams):
    ngrams_freq = {}
    for ngram_key in ngrams.keys():
        length = len(ngram_key.split())
        ngrams_freq[length] = ngrams_freq[length] + 1 if length in \
            ngrams_freq else 1

    ngrams_freq_sorted = sorted(ngrams_freq.items(), key=lambda item: item[0])
    print(" Ngram frequencies: {}".format(ngrams_freq_sorted), flush=True)
    print(" Entities in ngrams {} min_ngram_size {} max_ngram_size {}".format(\
            len(ngrams), ngrams_freq_sorted[0][0], ngrams_freq_sorted[len(\
            ngrams_freq_sorted) -1 ][0]), flush=True)
    return ngrams_freq_sorted

def get_ngrams_below_threshold(args, ngrams, ngrams_below_threshold, \
    dedup_file, dedup_key, ngrams_freq_sorted):

    start_time = time.time()
    # get the ngrams frequency
    args.get_ngram_freq_only = True
 
    # Open the large file to process in parallel
    num_workers = args.num_threads 
    pool = multiprocessing.Pool(num_workers)
    fin = open(dedup_file, 'r', encoding='utf-8')
    free_ngram_abt_partial=partial(free_ngram, args=args, key=dedup_key, \
        ngrams=ngrams, ngrams_freq_sorted=ngrams_freq_sorted)
    free_ngrams_abt = pool.imap(free_ngram_abt_partial, fin, 500)
 
    counter = 0
    for _, _, _, local_ngram in free_ngrams_abt:
        counter += 1
        if counter % 1000 == 0:
            print(' [compute_stat]> processed {} documents in {:.2f} seconds ...'.
                    format(counter, time.time() - start_time), flush=True)
        for local_key in local_ngram:
            if local_key in ngrams:
                ngrams[local_key] += 1
        local_ngram = {}

    print(' Time taken to compute statistics {:.2f} seconds'.format(time.time() - \
        start_time), flush=True)
    pool.close()
    pool.join()

    start_time = time.time()
    counter_threshold = 0
    # Get ngram below theadhold
    for local_key, local_val in ngrams.items():
        if ngrams[local_key] < args.key_threshold:
            print(" [threshold] {} {}".format(local_key, local_val), flush=True)
            counter_threshold += 1
            ngrams_below_threshold[local_key] = 1
            
    print(' Ngrams below threshold {}'.format(counter_threshold), flush=True)
    fin.close()

def clean_ngrams_below_threshold(args, ngrams_below_threshold, dedup_file, \
    dedup_key):

    start_time = time.time()
    # Now actually filter the dataset
    args.get_ngram_freq_only = False
    #id_prefix = '-'.join(args.tasks[::2])
    id_prefix = '-'.join(args.tasks[::1])

    # get the range of the size of the ngrams
    ngrams_freq_sorted = compute_ngram_freq_sorted(args, ngrams_below_threshold)

    # Open the large file to process in parallel
    counter = splitted = ignored = split_mt_thld = trimmed_count = 0
    num_workers = args.num_threads
    pool = multiprocessing.Pool(num_workers)
    fin = open(dedup_file, 'r', encoding='utf-8')
    free_ngram_clean_partial=partial(free_ngram, args=args, key=dedup_key, \
        ngrams=ngrams_below_threshold, ngrams_freq_sorted=ngrams_freq_sorted)
    free_ngrams_clean = pool.imap(free_ngram_clean_partial, fin, 500)
 
    out_f = open(args.output, 'wb')

    for text_buf_ngram_free, trimmed, myjson, _ in free_ngrams_clean:
        counter += 1
        try:

            trimmed_count += trimmed

            if len(text_buf_ngram_free) > 1:
                splitted += 1
            if len(text_buf_ngram_free) == 0:
                ignored += 1
            # more than 10 splits ignored
            if len(text_buf_ngram_free) > args.splits_count:
                text_buf_ngram_free = []
                split_mt_thld += 1

            if args.output is not None:
                if "split_id" in myjson:
                    use_prefix = myjson["split_id"] + "-"
                else:
                    use_prefix = ""

                for i in range(len(text_buf_ngram_free)):
                    split_id_string = id_prefix + '-{:010d}'.format(int(\
                        counter)) + '-{:04d}'.format(int(i))
                    myjson[dedup_key] = text_buf_ngram_free[i]
                    myjson["split_id"] = use_prefix + split_id_string
                    outjson = json.dumps(myjson, ensure_ascii=False)
                    #outjson = json.dumps({"text":text_buf_ngram_free[i],
                    #    id_prefix+"_split_id":split_id_string},
                    #    ensure_ascii=False)
                    out_f.write(outjson.encode('utf-8'))
                    out_f.write('\n'.encode('utf-8'))

            if counter % 1000 == 0:
                print(' [final]> processed {} documents in {:.2f} seconds ...'.
                    format(counter, time.time() - start_time), flush=True)
        except Exception as e:
            print('Error:', e)

    print(' [final]> processed {} documents in {:.2f} seconds ...'.
        format(counter, time.time() - start_time), flush=True)
    
    print(' Total docs {} splitted {} ignored {} splits > theshold {} trimmed'\
        ' {}'.format(counter, splitted, ignored, split_mt_thld, trimmed_count)\
        , flush=True)

    pool.close()
    pool.join()

    out_f.close()
    fin.close()

if __name__ == '__main__':

    # we use 13-grams, any text less than 200 characters got removed
    # any text splitted more than 10 got removed as well

    print('parsing the arguments ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs = '*', required=True, default=None, \
                        help = 'Tasks to use for deduplication: currently '
                        ' suuport [lambada, squad, natural_questions,'
                        ' triviaqa, webqa, race, drop, coqa, and piqa]')
    parser.add_argument('--lambada-path', type=str, default=None,
                       help='Only Lambada task needs the path')
    parser.add_argument('--dedup-dataset', nargs = '*', default=None,
                       help='Dataset to deduplicate with the key to use'
                        ' e.g. cc.json text')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name to save dedup dataset')
    parser.add_argument('--num-threads', type=int, default=40,
                       help='Number of threads to use')
    # Default dedup values
    parser.add_argument('--max-ngram-size', type=int, default=13,
                       help='Maximum size of ngram to use.')
    parser.add_argument('--min-ngram-size', type=int, default=8,
                       help='Minimum size of ngram to use.')
    parser.add_argument('--filter-text-char-len', type=int, default=200,
                       help='Remove any text below this length.')
    parser.add_argument('--key-threshold', type=int, default=10,
                       help='Number of keys to consider as threshold')
    parser.add_argument('--save-dictionary', type=str, default=None,
                       help='Save the dictionary')
    parser.add_argument('--load-dictionary', type=str, default=None,
                       help='Load the dictionary')
    parser.add_argument('--splits-count', type=int, default=10,
                       help='Remove any documents more than this many splits')
    parser.add_argument('--remove-char-each-side', type=int, default=200,
                       help='Maximum size of ngram to use.')

    args = parser.parse_args()

    assert len(args.dedup_dataset) == 2
    dedup_file = args.dedup_dataset[0]
    dedup_key = args.dedup_dataset[1]

    # Setup multi-processing
    num_workers = args.num_threads
    if args.load_dictionary is None:

        # Build ngrams
        ngrams = {}
        compute_tasks_ngrams(args, ngrams)

        # get the range of the size of the ngrams
        ngrams_freq_sorted = compute_ngram_freq_sorted(args, ngrams)

        # get ngram freq from large file in parallel
        # get ngrams below threshold
        ngrams_below_threshold = {}
        get_ngrams_below_threshold(args, ngrams, ngrams_below_threshold, \
            dedup_file, dedup_key, ngrams_freq_sorted)

        # save the dictionary if needed
        if args.save_dictionary is not None:
            with open(args.save_dictionary, 'wb') as save_dict_handle:
                pickle.dump(ngrams_below_threshold, save_dict_handle)
    else:
        with open(args.load_dictionary, 'rb') as load_dict_handle:
            ngrams_below_threshold = pickle.load(load_dict_handle)

    # filter the large file
    if args.output is not None:
        clean_ngrams_below_threshold(args, ngrams_below_threshold, \
            dedup_file, dedup_key)

    print('done :-)')
