# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
from datasets import load_dataset

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.generate import simple_generate
from megatron.post_training.model_provider import model_provider
from megatron.post_training.utils import report_current_memory_info
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.utils import print_rank_0, unwrap_model

warnings.filterwarnings('ignore')


def add_mmlu_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title='ModelOpt text generation ptq')
    group.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm.")
    group.add_argument("--percentage", type=float, default=1.0)
    group.add_argument("--lower-bound", type=float, default=None)
    add_modelopt_args(parser)
    return parser


def get_all_subjects():
    """Return all MMLU subjects."""
    return [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions',
    ]


def format_example(example, include_answer: bool = True):
    """Format an example into a multi-choices problem."""
    prompt = example["question"]
    for choice, answer in zip(["A", "B", "C", "D"], example["choices"]):
        prompt += "\n{}. {}".format(choice, answer)
    if include_answer:
        prompt += "Answer: {}\n\n".format(example["answer"])
    else:
        prompt += "\nAnswer:"
    return prompt


def generate_prompt(test_example, dev_examples, few_shots=0):
    """Generating few-shot prompts."""
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        " ".join(test_example["subject"].split("_"))
    )
    for i in range(few_shots):
        prompt += format_example(dev_examples[i])
    prompt += format_example(test_example, include_answer=False)
    return prompt


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_mmlu_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()

    disable_tqdm = args.disable_tqdm or torch.distributed.get_rank() > 0

    tokenizer = get_tokenizer()._tokenizer
    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    unwrapped_model = unwrap_model(model)[0]

    all_subjects = get_all_subjects()

    all_correct = {}

    for subject in all_subjects:
        test_data = load_dataset("cais/mmlu", subject, split="test")
        dev_data = load_dataset("cais/mmlu", subject, split="dev")

        correct = []
        for idx, test_example in enumerate(test_data):
            if idx > args.percentage * len(test_data):
                break
            prompt = generate_prompt(test_example, dev_data, few_shots=0)
            label = ["A", "B", "C", "D"][test_example["answer"]]
            tokens = tokenizer(prompt, return_tensors="pt")
            generated_ids = simple_generate(
                unwrapped_model, tokens.input_ids.cuda(), osl=2, disable_tqdm=disable_tqdm
            )
            predict = tokenizer.batch_decode(generated_ids)[0].strip()
            correct += [True] if predict.startswith(label) else [False]
        all_correct[subject] = correct

        if torch.distributed.get_rank() == 0:
            print(
                "{:48}| {:.3f} | {:5}/{:5}".format(
                    subject, sum(correct) / len(correct), sum(correct), len(correct)
                ),
                flush=True,
            )

    avg_correct = []

    for subject, correct in all_correct.items():
        avg_correct += correct

    if torch.distributed.get_rank() == 0:
        print(
            "{:48}| {:.3f} | {:5}/{:5}".format(
                "average", sum(avg_correct) / len(avg_correct), sum(avg_correct), len(avg_correct)
            ),
            flush=True,
        )

    if args.lower_bound is not None:
        assert sum(avg_correct) / len(avg_correct) > args.lower_bound
