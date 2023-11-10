# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
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


"""GPT sft dataset."""


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
            else:
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


def get_processed_dataset(name, data_folder):
    training_file = data_folder + "/{}/{}_QA_train*.json".format(name, name)
    validation_file = data_folder + "/{}/{}_QA_dev.json".format(name, name)

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
        self.dataset_name = name  # dataset_name equals to data_prefix in pretrain
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

    def __len__(self):
        return len(list(self.indexed_dataset))

    def __getitem__(self, idx):

        idx = idx % len(self.indexed_dataset)
        sample = self.indexed_dataset[idx]

        if self.args.retro_add_retriever:
            return build_retro_training_sample(sample,
                                               self.max_seq_length,  # needed for padding
                                               self.pad_id, self.eos_id,
                                               self.dataset_name,
                                               self.args.ft_neighbours,
                                               self.args.shuffle_topn)
        else:
            return build_normal_training_sample(sample,
                                                self.max_seq_length,  # needed for padding
                                                self.pad_id, self.eos_id,
                                                self.dataset_name,
                                                self.args.ft_neighbours,
                                                self.args.shuffle_topn,
                                                self.fewshot_list)


def reformat_prompt_retro(query, neighbours, dataset_name, ft_neighbours, \
                          max_output_len, tokenizer, max_seq_length):
    system = ("System: This is a chat between a user and an artificial intelligence assistant. The assistant gives "
              "helpful, detailed, and polite answers to the user's questions.\n\n")

    if dataset_name in ["oasst", "quiet_cockatoo", "open_inst", "quiet-cockatoo_commercial"]:
        input_tokens = tokenizer.tokenize(system + query)
        return input_tokens

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq",
                               "tqa", "quac"]
    yes_no_without_context = ["BoolQ"]
    multichoices = [""]
    formatted_dataset_name = ["doc2dial", "quac", "qrecc", "sharc"]

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
        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens) - len(system_tokens)]
        context = tokenizer.detokenize(context_tokens)

        all_input = system + context + dialogue_turn
        print(all_input)
        input_tokens = tokenizer.tokenize(all_input)
    else:
        all_input = system + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)

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


def reformat_prompt(query, neighbours, dataset_name, ft_neighbours, \
                    max_output_len, tokenizer, max_seq_length, template_id=0):
    system = ("System: This is a chat between a user and an artificial intelligence assistant. The assistant gives "
              "helpful, detailed, and polite answers to the user's questions based on the context. The assistant "
              "should also indicate when the answer cannot be found in the context.\n\n")

    if dataset_name in ["oasst", "quiet_cockatoo", "open_inst", "quiet-cockatoo_commercial"]:
        input_tokens = tokenizer.tokenize(system + query)
        return input_tokens

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq",
                               "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA", "tqa"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "quac", "qrecc", "sharc"]

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


def build_normal_training_sample(sample,
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

    input_tokens = reformat_prompt(query, neighbours, dataset_name, ft_neighbours, len(output_tokens), tokenizer,
                                   max_seq_length)

    # Padding
    tokens, answer_mask \
        = pad_and_convert_to_numpy(input_tokens, output_tokens,
                                   pad_id, max_seq_length, eos_id)

    train_sample = {
        'text': tokens,
        'answer_mask': answer_mask,
    }
    return train_sample


def build_retro_training_sample(sample,
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

    input_tokens = reformat_prompt_retro(query, neighbours, dataset_name, ft_neighbours, len(output_tokens), tokenizer,
                                         max_seq_length)

    # Padding
    tokens, answer_mask \
        = pad_and_convert_to_numpy(input_tokens, output_tokens,
                                   pad_id, max_seq_length, eos_id)

    # get retro neighbors
    args = get_args()
    retro_args = get_retro_args()
    n_chunks_per_sample = 2  # context chunk and answer chunk
    num_neighbors = args.retro_num_neighbors
    neighbor_tokens = np.zeros([n_chunks_per_sample, num_neighbors, retro_args.retro_gpt_retrieved_length],
                               dtype=np.int64)  # disable retro encoder

    train_sample = {
        'text': tokens,
        'answer_mask': answer_mask,
        'neighbor_tokens': neighbor_tokens,
        'context_len': len(input_tokens)
    }
    return train_sample



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
