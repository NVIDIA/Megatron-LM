#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#

# The following code has been taken from
# https://github.com/facebookresearch/DPR, which is CC-BY-NC 4.0
# licensed as of now. More details on the license can be found
# at https://github.com/facebookresearch/DPR/blob/master/LICENSE

"""
 Set of utilities for Q&A results validation tasks - Retriver passage
 validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

import regex as re
from tasks.orqa.unsupervised.tokenizers import SimpleTokenizer

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits',\
                                        'questions_doc_hits'])

def calculate_matches(all_docs: Dict[object, Tuple[str, str]], 
    answers: List[List[str]], closest_docs: List[Tuple[List[object], 
    List[float]]], workers_num: int, match_type: str) -> QAMatchStats:
    """
    Evaluates answers presence in the set of documents. This function is 
    supposed to be used with a large collection of documents and results. 
    It internally forks multiple sub-processes for evaluation and then 
    merges results
    :param all_docs: dictionary of the entire documents database. 
        doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their
        scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for
        available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved
        and the value is the total amount of valid matches across an entire
        dataset.
    questions_doc_hits - more detailed info with answer matches for every
        question and every retrieved document
    """
    global dpr_all_documents
    dpr_all_documents = all_docs

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(
        processes=workers_num,
    )

    logger.info('Matching answers in top docs...')

    get_score_partial = partial(check_answer, match_type=match_type,
                                    tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)

    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """
    Search through all the top docs to see if they have any of the answers.
    """
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        if has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """
    Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text 
        and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == 'string':
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    elif match_type == 'regex':
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except Exception:
        return False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    return unicodedata.normalize('NFD', text)
