# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Detokenization."""

import re


def ptb_detokenizer(string):
    string = string.replace(" '", "'")
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" n't", "n't")
    string = string.replace(" N ", "1 ")
    string = string.replace("$ 1", "$1")
    string = string.replace("# 1", "#1")
    return string


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def lambada_detokenizer(string):
    return string


_DETOKENIZERS = {
    'ptb': ptb_detokenizer,
    'wiki': wikitext_detokenizer,
    'lambada': lambada_detokenizer,
}


def get_detokenizer(path):
    for key in _DETOKENIZERS.keys():
        if key in path:
            return _DETOKENIZERS[key]
