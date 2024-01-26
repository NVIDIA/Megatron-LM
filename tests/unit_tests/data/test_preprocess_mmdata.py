# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import random
import sys
import tempfile

import nltk
import numpy

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from tests.unit_tests.data.test_preprocess_data import dummy_jsonl, gpt2_merge, gpt2_vocab
from tools.merge_datasets import main as merge_main
from tools.preprocess_mmdata import Encoder
from tools.preprocess_mmdata import get_args as build_args
from tools.preprocess_mmdata import main as build_main


def dummy_img(odir_txt, odir_img):
    for name in os.listdir(odir_txt):
        with open(os.path.join(odir_txt, name), "rt") as reader_txt:
            length = sum(1 for _ in reader_txt)
        os.makedirs(os.path.join(odir_img, os.path.splitext(name)[0]), exist_ok=False)
        for i in range(length):
            with open(
                os.path.join(odir_img, os.path.splitext(name)[0], f"{str(i).zfill(4)}.img"), "wb"
            ) as writer_img:
                # 32 * 32 - 1 to induce preprocessing 0-index padding
                writer_img.write(bytes([random.randint(0, 255) for _ in range(32 * 32 - 1)]))


def build_datasets(idir_txt, idir_img, odir, extra_args=[]):
    for name in os.listdir(idir_txt):
        sys.argv = [
            sys.argv[0],
            "--input",
            os.path.join(idir_txt, name),
            "--input-image",
            os.path.join(idir_img, os.path.splitext(name)[0]),
            "--output-prefix",
            os.path.join(odir, os.path.splitext(name)[0]),
        ] + extra_args
        build_main()


def merge_datasets(idir):
    sys.argv = [
        sys.argv[0],
        "--input",
        idir,
        "--output-prefix",
        os.path.join(idir, "merge"),
        "--multimodal",
    ]
    merge_main()


def do_test_preprocess_mmdata(temp_dir, extra_args=[]):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws_txt = os.path.join(temp_dir, "sample_raws_txt")
    path_to_raws_img = os.path.join(temp_dir, "sample_raws_img")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_raws_txt)
    os.mkdir(path_to_raws_img)
    os.mkdir(path_to_data)

    # create the dummy text resources
    dummy_jsonl(path_to_raws_txt)

    # create the dummy image resources
    dummy_img(path_to_raws_txt, path_to_raws_img)

    # build the datasets
    build_datasets(
        path_to_raws_txt, path_to_raws_img, path_to_data, extra_args=extra_args,
    )

    # merge the datasets
    merge_datasets(path_to_data)

    sys.argv = [
        sys.argv[0],
        "--input",
        None,
        "--input-image",
        None,
        "--output-prefix",
        None,
    ] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        for option in ["decode", "detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except AttributeError:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot `decode` or `detokenize`.")

    merged_index = 0
    merged_dataset = MMapIndexedDataset(os.path.join(path_to_data, "merge"), multimodal=True)

    # sorted to ensure ordering matches merged dataset
    basenames = sorted(
        [
            name
            for name in os.listdir(path_to_data)
            if name.endswith(".idx") and not name.startswith("merge")
        ]
    )

    # index into the merged document index
    merged_doc_index_index = 0

    for basename in basenames:
        realpath_raw_txt = os.path.join(path_to_raws_txt, f"{os.path.splitext(basename)[0]}.jsonl")
        realpath_raw_img = os.path.join(path_to_raws_img, os.path.splitext(basename)[0])
        realpath_doc = os.path.join(path_to_data, os.path.splitext(basename)[0])

        dataset_index = 0
        dataset = MMapIndexedDataset(realpath_doc, multimodal=True)

        merged_doc_idx = merged_dataset.document_indices[
            merged_doc_index_index : merged_doc_index_index + len(dataset.document_indices)
        ]
        merged_doc_idx = merged_doc_idx - merged_doc_idx[0]

        assert (
            dataset.document_indices == merged_doc_idx
        ).all(), f"ERROR: {basename.split('_')[:-2]}: merged dataset document indices mismatch"

        merged_doc_index_index += len(dataset.document_indices) - 1

        with open(realpath_raw_txt, "rt") as reader:
            for json_line, image_path in zip(
                reader,
                [
                    os.path.join(realpath_raw_img, basename)
                    for basename in os.listdir(realpath_raw_img)
                ],
            ):
                toks, image, length = encoder.encode((json_line, image_path))

                raw_text = tokens_to_string(toks)
                # reverse to account for preprocessing 0-index padding
                raw_image = image[::-1]

                processed_toks = dataset[dataset_index][0]
                assert dataset[dataset_index][1] == 0
                processed_text = tokens_to_string(processed_toks)

                processed_image = dataset[dataset_index + 1][0]
                assert dataset[dataset_index + 1][1] == 1
                # reverse to account for preprocessing 0-index padding
                processed_image = processed_image[::-1][0 : raw_image.size]

                assert (
                    raw_text == processed_text
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents (text) do not match"

                assert numpy.allclose(
                    raw_image, processed_image
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents (image) do not match"

                dataset_index += 2

                merged_toks = merged_dataset[merged_index][0]
                assert merged_dataset[merged_index][1] == 0
                merged_text = tokens_to_string(merged_toks)

                merged_image = merged_dataset[merged_index + 1][0]
                assert merged_dataset[merged_index + 1][1] == 1
                # reverse to account for preprocessing 0-index padding
                merged_image = merged_image[::-1][0 : raw_image.size]

                assert (
                    raw_text == merged_text
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents (text) do not match"

                assert numpy.allclose(
                    raw_image, merged_image
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents (image) do not match"

                merged_index += 2

        print(
            f"INFO: {''.join(basename.split('_')[:-2])}: raw, processed, and merged documents match!"
        )

    print("INFO: Success!")


def test_preprocess_mmdata():
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
            "--pad-length",
            "1024",
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            gpt2_vocab(temp_dir),
            "--merge-file",
            gpt2_merge(temp_dir),
            "--append-eod",
            "--workers",
            "10",
            "--log-interval",
            "1",
        ]

        do_test_preprocess_mmdata(temp_dir, extra_args=gpt_args)


if __name__ == "__main__":
    test_preprocess_mmdata()
