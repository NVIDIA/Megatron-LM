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

import json
import collections
from multiprocessing.sharedctypes import Value
import os
import torch
import numpy as np
import glob
from megatron import get_tokenizer, get_args, get_retro_args


def format_multichoice(multichoice_options):
    options_text = ["({}) {}".format(chr(ord('A') + i), option) for i, option in
                    zip(range(len(multichoice_options)), multichoice_options)]
    return "Choose one based on the following options: {}".format(" ".join(options_text))


def format_multichoice_question(question, multichoice_options):
    return "{}\n{}".format(question, format_multichoice(multichoice_options))


def format_answer(answer):
    return " {}".format(answer)


"""GPT ft dataset."""


def preprocess(data_file, inference_only=False, retrieved_neighbours=False, fix_newsqa=True):
    args = get_args()
    assert args.ft_neighbours > 0
    if args.longform_answer:
        nq_examples = []
        with open(data_file, "r") as f:
            for fn in f:
                nq_examples.append(json.loads(fn))
    else:
        nq_examples = []
        for my_data_file in sorted(glob.glob(data_file)):
            with open(my_data_file, "r", encoding='utf-8') as f:
                nq_examples.extend(json.load(f))

    data = []
    for instance in nq_examples:
        question = instance["question"]
        if 'qa_type' in instance and instance['qa_type'] == "multi_choice_qa":
            question = format_multichoice_question(question, instance["multichoice_options"])
        if args.bert_retriever_neighbours:
            contexts = instance["bert_pretrain_corpus_neighbours"]
            neighbours = ["source: " + ctx for ctx in contexts]
        else:
            if retrieved_neighbours:
                contexts = instance["ctxs"]
                neighbours = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in contexts]
            else:
                if "sub-paragraphs" in instance:
                    if type(instance["sub-paragraphs"]) == list:  # doc2dial:
                        neighbours = [
                            "title: " + instance["sub-paragraphs"][0] + ", source: " + instance["sub-paragraphs"][1]]
                    else:
                        neighbours = ["title: , source: " + instance["sub-paragraphs"]]
                elif fix_newsqa and "sub_paragraph" in instance:
                    neighbours = ["title: , source: " + instance["sub_paragraph"]]
                else:
                    neighbours = ["title: , source: "]

        if inference_only:
            data.append((question, None, neighbours))
        else:
            if args.longform_answer:
                if "longform_answer" in instance:
                    answers = [instance["longform_answer"]]
                else:
                    continue
            else:
                if "answers" in instance:
                    answers = instance["answers"]
                elif "answer" in instance:
                    if type(instance["answer"]) is str:
                        answers = [instance["answer"]]
                    elif type(instance["answer"]) is list:
                        answers = instance["answer"]
                    else:
                        answers = [str(instance["answer"])]
                else:
                    raise ValueError("need to have answer or answers")
            if len(answers) < 1:
                continue
                # answers = ["This question cannot be answered based on the given information."]
            else:
                ## only take answer 0
                if type(answers[0]) is dict:
                    answers = [answers[0]["text"].strip()]
                elif type(answers[0]) is str:
                    answers = [answers[0]]
                else:
                    raise ValueError("unsupported type for answer(s)")

            for answer in answers:
                answer = format_answer(answer)
                data.append((question, answer, neighbours))

    return data


def eli5_preprocess(data_file):
    eli5_examples = []
    with open(data_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            eli5_examples.append(json.loads(line))

    data = []
    for i, d in enumerate(eli5_examples):
        if "output" not in d or "input" not in d:
            continue
        answer = None
        neighbours = None
        question = d["input"]
        if "neighbours" in d:
            neighbours = d["neighbours"]

        for item in d["output"]:
            if "answer" in item:
                answer = item["answer"]
                data.append((question, answer, neighbours))
                # if "provenance" in item:
            #     if len(item["provenance"]) > 1:
            #         print(i, "more than one")
            #     print("found provenance", item["provenance"], "\n")
    return data


def load_incontext_fewshot_samples(data_file, n_shot):
    with open(data_file, "r") as f:
        data_list = json.load(f)

    assert len(data_list) >= n_shot
    data_list = data_list[:n_shot]

    return data_list


def get_processed_dataset(name, data_folder, processed=True, ratio=None, index=None, num_samples=None):
    if name.lower() == 'eli5':
        if processed:
            training_file = data_folder + "/eli5-train-kilt-with-neighbours.jsonl"
            validation_file = data_folder + "/eli5-dev-kilt-with-neighbours.jsonl"
            test_file = data_folder + "/eli5-test_without_answers-kilt.jsonl"
        else:
            training_file = data_folder + "/eli5-train-kilt.jsonl"
            validation_file = data_folder + "/eli5-dev-kilt.jsonl"
            test_file = data_folder + "/eli5-test_without_answers-kilt.jsonl"

        dataset = {}
        dataset["train"] = eli5_preprocess(training_file)
        dataset["valid"] = eli5_preprocess(validation_file)
        dataset["test"] = eli5_preprocess(test_file)
    else:

        training_file = data_folder + "/{}/{}_QA_train*.json".format(name, name)
        validation_file = data_folder + "/{}/{}_QA_dev.json".format(name, name)
        # test_file = data_folder + "/{}/{}_QA_test.json"

        dataset = {}
        dataset["train"] = preprocess(training_file)
        dataset["valid"] = preprocess(validation_file)
        dataset["test"] = preprocess(validation_file)

    print(name, "train", len(dataset["train"]))
    print(name, "valid", len(dataset["valid"]))
    print(name, "test", len(dataset["test"]))

    return dataset


def count_stat(dataset, tokenizer):
    args = get_args()
    nb_lens = []
    for i, d in enumerate(dataset):
        query, answer, neighbours = d
        nb_lens.extend([len(tokenizer.tokenize(neighbour)) for neighbour in neighbours[:args.k]])

    print("len of nb", len(nb_lens))
    print("max of len nb", max(nb_lens))
    print("num of cut ", sum([l > 128 for l in nb_lens]), sum([l > 128 for l in nb_lens]) // len(nb_lens))
    print("last max", sorted(nb_lens)[-10:])


class FtDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, max_seq_length,
                 max_seq_length_dec=0, fewshot_list=None):

        # Params to store.
        self.dataset_name = name  ## dataset_name equals to data_prefix in pretrain
        self.max_seq_length = max_seq_length
        self.desc = name

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.eos_id = tokenizer.eod
        self.pad_id = tokenizer.eod
        self.fewshot_list = fewshot_list

        self.args = get_args()

        # count_stat(indexed_dataset, tokenizer)

    def __len__(self):
        return len(list(self.indexed_dataset))

    def __getitem__(self, idx):

        idx = idx % len(self.indexed_dataset)
        sample = self.indexed_dataset[idx]

        if self.args.retro_add_retriever:
            return build_retro_training_sample_v2(sample,
                                                  self.max_seq_length,  # needed for padding
                                                  self.pad_id, self.eos_id,
                                                  self.dataset_name,
                                                  self.args.ft_neighbours,
                                                  self.args.shuffle_topn)
        else:
            return build_normal_training_sample_v2(sample,
                                                   self.max_seq_length,  # needed for padding
                                                   self.pad_id, self.eos_id,
                                                   self.dataset_name,
                                                   self.args.ft_neighbours,
                                                   self.args.shuffle_topn,
                                                   self.fewshot_list)


def reformat_prompt_v1(query, neighbours, dataset_name, ft_neighbours, \
                       max_output_len, tokenizer, max_seq_length):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"

    if dataset_name in ["oasst", "quiet_cockatoo"]:
        input_tokens = tokenizer.tokenize(system + query)
        # print(dataset_name, system + query)
        return input_tokens

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq",
                               "tqa", "quac"]
    yes_no_without_context = ["BoolQ"]
    multichoices = [""]
    formatted_dataset_name = ["doc2dial", "quac", "qrecc", "sharc"]
    user_template = ""

    ## fix bug format for formatted text, no change
    if dataset_name in formatted_dataset_name:
        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            user = "{} Answer the above question with a short phrase.".format(query)
        elif dataset_name in yes_no_without_context:
            user = "{} Answer the above question with True or False.".format(query)
        else:
            user = "{} Answer the above question with a long complete answer.".format(query)

        if dataset_name in short_span_with_context:
            dialogue_format = "User: {}\n\nAssistant: The answer is"
            dialogue_turn = dialogue_format.format(user)
        else:
            dialogue_format = "User: {}\n\nAssistant:"
            dialogue_turn = dialogue_format.format(user)

    if ft_neighbours > 0:
        # if shuffle_topn:
        #     import random
        #     random.seed(1234)
        #     random_neighbours = neighbours[0:ft_neighbours]
        #     random.shuffle(random_neighbours)
        #     neighbours = random_neighbours + neighbours[ft_neighbours:]
        # Truncate to `max_sequence_length` to fit in output tokens.
        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens) - len(system_tokens)]
        context = tokenizer.detokenize(context_tokens)

        all_input = system + context + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)
    else:
        all_input = system + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)

    # print(dataset_name, all_input)

    return input_tokens


def flan_format(system, context, dialogue_turn, template_id=0):
    templates = [
        "{}User: Answer based on context:\n\n{}{}",
        "{}User: {}Answer this question based on the article: {}",
        "{}User: {}{}",
        "{}User: {}Answer this question: {}",
        "{}User: Read this article and answer this question {}{}",
        "{}User: {}Based on the above article, answer a question. {}",
        "{}User: Context: {}Question: {}"
    ]
    template = templates[template_id - 1].format(system, context, dialogue_turn)
    return template


def reformat_prompt_v2(query, neighbours, dataset_name, ft_neighbours, \
                       max_output_len, tokenizer, max_seq_length, template_id=0):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n"

    if dataset_name in ["oasst", "quiet_cockatoo"]:
        input_tokens = tokenizer.tokenize(system + query)
        # print(dataset_name, system + query)
        return input_tokens

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq",
                               "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA", "tqa"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "quac", "qrecc", "sharc"]
    user_template = ""

    ## fix bug format for formatted text, no change
    if dataset_name in formatted_dataset_name:
        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            if template_id == 0:
                user = "Answer the following question with a short span. {}".format(query)
            else:
                user = query
        elif dataset_name in yes_no_without_context:
            user = "Answer the following question with True or False. {}".format(query)
        elif dataset_name in multichoices:
            user = "Answer the following question by selecting one of the provided options. {}".format(query)
        else:
            if template_id == 0:
                user = "Please give a full and complete answer for the question. {}".format(query)
            else:
                user = query

        if dataset_name in short_span_with_context:
            if template_id == 0:
                dialogue_format = "User: {}\n\nAssistant: The answer is"
            else:
                dialogue_format = "{}\n\nAssistant: The answer is"
            dialogue_turn = dialogue_format.format(user)
        else:
            if template_id == 0:
                dialogue_format = "User: {}\n\nAssistant:"
            else:
                dialogue_format = "{}\n\nAssistant:"
            dialogue_turn = dialogue_format.format(user)

    if ft_neighbours > 0:
        # if shuffle_topn:
        #     import random
        #     random.seed(1234)
        #     random_neighbours = neighbours[0:ft_neighbours]
        #     random.shuffle(random_neighbours)
        #     neighbours = random_neighbours + neighbours[ft_neighbours:]
        # Truncate to `max_sequence_length` to fit in output tokens.
        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens) - len(system_tokens)]
        context = tokenizer.detokenize(context_tokens)

        if template_id == 0:
            all_input = system + context + dialogue_turn
        else:
            all_input = flan_format(system, context, dialogue_turn, template_id=template_id)
        input_tokens = tokenizer.tokenize(all_input)
    else:
        all_input = system + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)

    # print(dataset_name, all_input)

    return input_tokens


def reformat_prompt_short(query, neighbours, dataset_name, ft_neighbours, \
                       max_output_len, tokenizer, max_seq_length):

    if not query.endswith("?"):
        query = query + "?"
    query = "Question: {} Answer: The answer is".format(query)

    if ft_neighbours > 0:
        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(query)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens)]
        context = tokenizer.detokenize(context_tokens)
        all_input = context + query
        input_tokens = tokenizer.tokenize(all_input)
    else:
        all_input = query
        input_tokens = tokenizer.tokenize(all_input)

    return input_tokens


def reformat_prompt_with_fewshot_samples(query, neighbours, dataset_name, ft_neighbours, fewshot_list, \
                                         max_output_len, tokenizer, max_seq_length, multiturn_max_fewshot=3):
    # system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n"

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq",
                               "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "quac", "qrecc", "sharc"]
    user_template = ""

    if dataset_name in formatted_dataset_name:
        instruction = None
        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            # user = "Answer the following question with a short span. {}".format(query)
            instruction = "Answer the following question with a short span."
            user = instruction + " " + query
        elif dataset_name in yes_no_without_context:
            # user = "Answer the following question with True or False. {}".format(query)
            instruction = "Answer the following question with True or False."
            user = instruction + " " + query
        elif dataset_name in multichoices:
            instruction = "Answer the following question by selecting one of the provided options."
            user = instruction + " " + query
        else:
            # user = "Please give a full and complete answer for the question. {}".format(query)
            instruction = "Please give a full and complete answer for the question."
            user = instruction + " " + query

        dialogue_format = "User: {}\n\nAssistant:"
        dialogue_turn = dialogue_format.format(user)

    multiturn_dataset_name = formatted_dataset_name + ["quiet_cockatoo"]
    if dataset_name in multiturn_dataset_name:
        fewshot_list = fewshot_list[:multiturn_max_fewshot]

    fewshot_prompt = "Here are some question answer samples between user and assistant:\n\n"
    for i, item in enumerate(fewshot_list):
        question = item['question']
        answer = item['answer']
        if question.endswith("\n\nAssistant:"):
            assert instruction is None
            formatted_sample = question + " " + answer
        else:
            assert instruction is not None
            formatted_sample = "User: " + instruction + " " + question + "\n\nAssistant: " + answer

        fewshot_prompt += "Sample %d:\n\n" % (i + 1)
        fewshot_prompt += formatted_sample + "\n\n"
    fewshot_prompt += "Assistant should follow the answer formats from the aboved samples and give a response to the following user's question.\n\n"

    if dataset_name in ["oasst", "quiet_cockatoo"]:
        # input_tokens = tokenizer.tokenize(system + query)
        input_tokens = tokenizer.tokenize(system + fewshot_prompt + query)
        # print(dataset_name, system + query)
        return input_tokens

    if ft_neighbours > 0:
        # if shuffle_topn:
        #     import random
        #     random.seed(1234)
        #     random_neighbours = neighbours[0:ft_neighbours]
        #     random.shuffle(random_neighbours)
        #     neighbours = random_neighbours + neighbours[ft_neighbours:]
        # Truncate to `max_sequence_length` to fit in output tokens.
        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        fewshot_tokens = tokenizer.tokenize(fewshot_prompt)
        context_tokens = context_tokens[
                         :max_seq_length - max_output_len - len(dialogue_tokens) - len(fewshot_tokens) - len(
                             system_tokens)]
        context = tokenizer.detokenize(context_tokens)

        ## already try to put fewshot_prompt between system and context, results are not good
        all_input = system + context + fewshot_prompt + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)
    else:
        all_input = system + fewshot_prompt + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)

    # print(dataset_name, all_input)

    return input_tokens


def build_normal_training_sample_v2(sample,
                                    max_seq_length,
                                    pad_id,
                                    eos_id,
                                    dataset_name,
                                    ft_neighbours=1,
                                    shuffle_topn=False,
                                    fewshot_list=None):
    # unpack tokens
    query, answer, neighbours = sample

    # tokenization
    tokenizer = get_tokenizer()
    output_tokens = tokenizer.tokenize(answer)

    # input_tokens = reformat_prompt_v1(query, neighbours, dataset_name, ft_neighbours, len(output_tokens), tokenizer, max_seq_length)
    input_tokens = reformat_prompt_v2(query, neighbours, dataset_name, ft_neighbours, len(output_tokens), tokenizer,
                                      max_seq_length)
    # print(answer)

    # print(repr(tokenizer.detokenize(input_tokens)), repr(tokenizer.detokenize(output_tokens)), dataset_name)
    # Padding
    tokens, answer_mask \
        = pad_and_convert_to_numpy(input_tokens, output_tokens,
                                   pad_id, max_seq_length, eos_id)

    train_sample = {
        'text': tokens,
        'answer_mask': answer_mask,
    }
    return train_sample


def build_retro_training_sample_v2(sample,
                                   max_seq_length,
                                   pad_id,
                                   eos_id,
                                   dataset_name,
                                   ft_neighbours=1,
                                   shuffle_topn=False):
    # unpack tokens
    query, answer, neighbours = sample

    # tokenization
    tokenizer = get_tokenizer()
    output_tokens = tokenizer.tokenize(answer)

    input_tokens = reformat_prompt_v1(query, neighbours, dataset_name, ft_neighbours, len(output_tokens), tokenizer,
                                      max_seq_length)
    # print(answer)

    # print(repr(tokenizer.detokenize(input_tokens)), repr(tokenizer.detokenize(output_tokens)), dataset_name)
    # Padding
    tokens, answer_mask \
        = pad_and_convert_to_numpy(input_tokens, output_tokens,
                                   pad_id, max_seq_length, eos_id)

    # get retro neighbors
    args = get_args()
    retro_args = get_retro_args()
    n_chunks_per_sample = 2
    num_neighbors = args.retro_num_neighbors
    neighbor_tokens = np.zeros([n_chunks_per_sample, num_neighbors, retro_args.retro_gpt_retrieved_length],
                               dtype=np.int64)
    # print("neighbor_tokens.shape", neighbor_tokens.shape)

    train_sample = {
        'text': tokens,
        'answer_mask': answer_mask,
        'neighbor_tokens': neighbor_tokens,
        'context_len': len(input_tokens)
    }
    return train_sample


def build_retro_training_sample(sample,
                                max_seq_length,
                                pad_id,
                                eos_id,
                                dataset_name,
                                ft_neighbours=1):
    """Build training sample for retro NQ.
    """

    # unpack tokens
    query, answer, neighbours = sample
    assert neighbours is not None

    # tokenization
    tokenizer = get_tokenizer()
    input_tokens = tokenizer.tokenize(query)
    output_tokens = tokenizer.tokenize(answer)

    # prompt learning to add soft token place holders
    args = get_args()

    if dataset_name == 'eli5':
        # print(len(output_tokens), args.m, num_samples, len(c_answers))
        nb_tokens = [[tokenizer.tokenize(dpr_neighhour_i) for dpr_neighhour_i in dpr_neighbour] for dpr_neighbour in
                     neighbours]
    else:
        if args.question_in_encoder:
            neighbours = ["question: {}, ".format(query) + neighbour if i >= ft_neighbours else neighbour for
                          i, neighbour in enumerate(neighbours)]
            nb_tokens = [tokenizer.tokenize(neighbour) for neighbour in neighbours]
        if args.prefix:
            neighbours = ["Evidence {} ".format(i) + neighbour if i >= ft_neighbours else neighbour for i, neighbour in
                          enumerate(neighbours)]
            # print(neighbours[0])
            nb_tokens = [tokenizer.tokenize(neighbour) for neighbour in neighbours]
        else:
            nb_tokens = [tokenizer.tokenize(neighbour) for neighbour in neighbours]
    # elif dataset_name == 'nq' or dataset_name == 'tqa':

    if ft_neighbours > 0:
        # Truncate to `max_sequence_length` to fit in output tokens.
        ## most relevant nb should be the last
        context = "\n".join(neighbours[0:ft_neighbours][::-1]) + "\n"
        context_tokens = tokenizer.tokenize(context)
        ## truncate the beginning tokens
        context_tokens = context_tokens[-(max_seq_length - args.m - len(input_tokens)):]
        input_tokens = context_tokens + input_tokens

    # Left pad input tokens to args.m
    input_tokens = left_pad_question(args, input_tokens, pad_id)
    # input_tokens = input_tokens[:args.m]
    # left_pad_len = args.m - len(input_tokens)
    # input_tokens = [pad_id] * left_pad_len + input_tokens

    # Padding
    tokens, answer_mask \
        = pad_and_convert_to_numpy(input_tokens, output_tokens,
                                   pad_id, max_seq_length, eos_id)

    # take top k neighbours and padding
    if dataset_name == 'eli5':
        neighbours_tokens = pad_neighbours_for_q_and_a(args, nb_tokens, pad_id)
    else:
        neighbours_tokens = pad_neighbours_for_query_only(args, nb_tokens, pad_id, ft_neighbours)
    # elif dataset_name == 'nq' or dataset_name == 'tqa':
    # neighbours_tokens = []
    # for nb_token in nb_tokens[:args.k]:
    #     if len(nb_token) >= args.r:
    #         nb_token = nb_token[:args.r]
    #     else:
    #         nb_token =  nb_token + [pad_id] * (args.r - len(nb_token))
    #     neighbours_tokens.append(nb_token)
    # if len(neighbours_tokens) < args.k:
    #     assert ValueError("neighbours are not enough, to do: add empty ones and create mask for those empty ones")
    # neighbours_tokens = np.array(neighbours_tokens).reshape(1, args.k, args.r).repeat(args.seq_length / args.m, axis=0) ## dim (l, k, r)

    train_sample = {
        'text': tokens,
        'answer_mask': answer_mask,
        'neighbor_tokens': neighbours_tokens
    }
    return train_sample


def left_pad_question(args, input_tokens, pad_id):
    ## up padding to nearest m times n
    padded_len = args.m * (int((len(input_tokens) - 0.5) / args.m) + 1)
    left_pad_len = padded_len - len(input_tokens)
    assert left_pad_len >= 0
    input_tokens = [pad_id] * left_pad_len + input_tokens
    return input_tokens


def pad_neighbours_for_query_only(args, nb_tokens, pad_id, ft_neighbours):
    # take top k neighbours and padding
    neighbours_tokens = []

    if args.reuse_top:
        valid_nb_tokens = nb_tokens[:args.k]
    else:
        valid_nb_tokens = nb_tokens[ft_neighbours:args.k + ft_neighbours]

    for nb_token in valid_nb_tokens:
        if len(nb_token) >= args.r:
            # print("max len is {}, and the current one is {}".format(args.r, len(nb_token)))
            nb_token = nb_token[:args.r]
        else:
            nb_token = nb_token + [pad_id] * (args.r - len(nb_token))
        neighbours_tokens.append(nb_token)
    if len(neighbours_tokens) < args.k:
        assert ValueError("neighbours are not enough, to do: add empty ones and create mask for those empty ones")
    neighbours_tokens = np.array(neighbours_tokens).reshape(1, args.k, args.r).repeat(args.seq_length / args.m,
                                                                                      axis=0)  ## dim (l, k, r)
    return neighbours_tokens


def pad_neighbours_for_q_and_a(args, nb_tokens, pad_id):
    # take top k neighbours and padding
    neighbours_tokens = []
    for nb_tokens_i in nb_tokens:
        neighbour_i_tokens = []
        assert len(nb_tokens_i) == args.k  ## top k retreived neighours
        for nb_token in nb_tokens_i:
            if len(nb_token) >= args.r:
                nb_token = nb_token[:args.r]
            else:
                nb_token = nb_token + [pad_id] * (args.r - len(nb_token))
            neighbour_i_tokens.append(nb_token)
        neighbours_tokens.append(neighbour_i_tokens)
    neighbours_tokens = np.array(neighbours_tokens)

    # dim (l, k, r)
    l = int(args.seq_length / args.m)
    if neighbours_tokens.shape[0] < l:
        neighbours_tokens = np.concatenate([neighbours_tokens,
                                            neighbours_tokens[-1:].repeat(l - neighbours_tokens.shape[0], axis=0)],
                                           axis=0)
    else:
        neighbours_tokens = neighbours_tokens[:l]

    return neighbours_tokens


def pad_and_convert_to_numpy(input_ids, output_ids,
                             pad_id, max_seq_length,
                             eos_id):
    """Pad sequences and convert them to numpy."""
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length - 1]

    if len(input_ids + output_ids) > max_seq_length:
        output_ids = output_ids[:max_seq_length - len(input_ids)]

    tokens = input_ids + output_ids
    answer_mask = [0] * len(input_ids) + [1] * len(output_ids)

    # padding
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0

    # Tokens.
    filler = [pad_id] * padding_length
    tokens = np.array(tokens + [eos_id] + filler, dtype=np.int64)

    # answer mask
    answer_mask = answer_mask + [1] + [0] * padding_length
    answer_mask = np.array(answer_mask, dtype=np.int64)

    return tokens, answer_mask
