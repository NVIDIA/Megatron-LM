# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import re
import json
import os
from typing import Any, Iterable, Dict

from numpy import ndarray
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.utils import Split
import torch
import numpy
import glob
from collections import OrderedDict

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from dataclasses import dataclass


_DATASET_NAME_PATTERNS = {
    Split.train: r"(?P<name>[^\0]+)\/(?P=name)\_QA\_train.json",
    Split.valid: r"(?P<name>[^\0]+)\/(?P=name)\_QA\_dev.json",
}


@dataclass
class JsonQADatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for the QA finetuning pipeline
    """
    ft_neighbours: int = 1

    bert_retriever_neighbours: bool = False

    longform_answer: bool = False

    inference_only: bool = False

    retrieved_neighbours: bool = False

    fix_newsqa: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.blend_per_split is not None


@dataclass
class RetroJsonQADatasetConfig(JsonQADatasetConfig):
    """Configuration object for the Retro QA finetuning pipeline
    """
    retro_num_neighbors: int = None

    retro_gpt_retrieved_length: int = None

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.retro_num_neighbors is not None
        assert self.retro_gpt_retrieved_length is not None


class JsonQADataset(MegatronDataset):

    def __init__(self, dataset: Any, dataset_path: str, indices: ndarray, num_samples: int, index_split: Split, config: BlendedMegatronDatasetConfig) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        matches = re.findall(_DATASET_NAME_PATTERNS[index_split], dataset_path)
        assert len(matches) == 1
        assert len(matches[0]) > 0
        self.dataset_name = matches[0]

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: JsonQADatasetConfig) -> Iterable:
        assert os.path.isfile(dataset_path), f"{dataset_path} does not exist on disk"
        return preprocess(dataset_path, config)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, ndarray]:
        sample = self.dataset[idx % len(self.dataset)]

        # unpack tokens
        query, answer, neighbours = sample

        # tokenization
        output_tokens = self.config.tokenizer.tokenize(answer)

        input_tokens = reformat_prompt(
            query,
            neighbours,
            self.dataset_name,
            self.config.ft_neighbours,
            len(output_tokens),
            self.config.tokenizer,
            self.config.sequence_length
        )

        # padding
        tokens, answer_mask = pad_and_convert_to_numpy(
            input_tokens, output_tokens, self.config.tokenizer.pad, self.config.sequence_length, self.config.tokenizer.eos
        )

        train_sample = {
            'text': tokens,
            'answer_mask': answer_mask,
        }

        return train_sample


class RetroJsonQADataset(JsonQADataset):

    def __getitem__(self, idx: int) -> Dict[str, ndarray]:

        sample = self.dataset[idx % len(self.dataset)]

        # unpack tokens
        query, answer, neighbours = sample

        # tokenization
        output_tokens = self.config.tokenizer.tokenize(answer)

        input_tokens = reformat_prompt_retro(
            query,
            neighbours,
            self.dataset_name,
            self.config.ft_neighbours,
            len(output_tokens),
            self.config.tokenizer,
            self.config.sequence_length
        )

        # padding
        tokens, answer_mask = pad_and_convert_to_numpy(
            input_tokens,
            output_tokens,
            self.config.tokenizer.pad,
            self.config.sequence_length,
            self.config.tokenizer.eos
        )

        # get retro neighbors
        # context chunk and answer chunk
        n_chunks_per_sample = 2
        num_neighbors = self.config.retro_num_neighbors
        # disable retro encoder
        neighbor_tokens = numpy.zeros(
            [n_chunks_per_sample, num_neighbors, self.config.retro_gpt_retrieved_length],
            dtype=numpy.int64
        )

        train_sample = {
            'text': tokens,
            'answer_mask': answer_mask,
            'neighbor_tokens': neighbor_tokens,
            'context_len': len(input_tokens)
        }

        return train_sample


def format_multichoice(multichoice_options):
    options_text = ["({}) {}".format(chr(ord('A') + i), option) for i, option in
                    zip(range(len(multichoice_options)), multichoice_options)]
    return "Choose one based on the following options: {}".format(" ".join(options_text))


def format_multichoice_question(question, multichoice_options):
    return "{}\n{}".format(question, format_multichoice(multichoice_options))


def format_answer(answer):
    return " {}".format(answer)


def preprocess(dataset_path: str, config: JsonQADatasetConfig):
    assert config.ft_neighbours > 0
    if config.longform_answer:
        nq_examples = []
        with open(dataset_path, "r") as f:
            for fn in f:
                nq_examples.append(json.loads(fn))
    else:
        nq_examples = []
        for my_data_file in sorted(glob.glob(dataset_path)):
            with open(my_data_file, "r", encoding='utf-8') as f:
                nq_examples.extend(json.load(f))

    data = []
    for instance in nq_examples:
        question = instance["question"]
        if 'qa_type' in instance and instance['qa_type'] == "multi_choice_qa":
            question = format_multichoice_question(question, instance["multichoice_options"])
        if config.bert_retriever_neighbours:
            contexts = instance["bert_pretrain_corpus_neighbours"]
            neighbours = ["source: " + ctx for ctx in contexts]
        else:
            if config.retrieved_neighbours:
                contexts = instance["ctxs"]
                neighbours = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in contexts]
            else:
                if "sub-paragraphs" in instance:
                    if type(instance["sub-paragraphs"]) == list:  # doc2dial:
                        neighbours = [
                            "title: " + instance["sub-paragraphs"][0] + ", source: " + instance["sub-paragraphs"][1]]
                    else:
                        neighbours = ["title: , source: " + instance["sub-paragraphs"]]
                elif config.fix_newsqa and "sub_paragraph" in instance:
                    neighbours = ["title: , source: " + instance["sub_paragraph"]]
                else:
                    neighbours = ["title: , source: "]

        if config.inference_only:
            data.append((question, None, neighbours))
        else:
            if config.longform_answer:
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


def count_stat(dataset, tokenizer, k):
    nb_lens = []
    for i, d in enumerate(dataset):
        query, answer, neighbours = d
        nb_lens.extend([len(tokenizer.tokenize(neighbour)) for neighbour in neighbours[:k]])

    print("len of nb", len(nb_lens))
    print("max of len nb", max(nb_lens))
    print("num of cut ", sum([l > 128 for l in nb_lens]), sum([l > 128 for l in nb_lens]) // len(nb_lens))
    print("last max", sorted(nb_lens)[-10:])


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
    tokens = numpy.array(tokens + [eos_id] + filler, dtype=numpy.int64)

    # answer mask
    answer_mask = answer_mask + [1] + [0] * padding_length
    answer_mask = numpy.array(answer_mask, dtype=numpy.int64)

    return tokens, answer_mask
