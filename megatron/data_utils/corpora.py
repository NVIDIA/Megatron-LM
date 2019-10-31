# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""several datasets with preset arguments"""
from .datasets import json_dataset, csv_dataset

class wikipedia(json_dataset):
	"""
	dataset for wikipedia with arguments configured for convenience

	command line usage: `--train-data wikipedia`
	"""
	#PATH = '/home/universal-lm-data.cosmos549/datasets/wikipedia/wikidump_lines.json'
	PATH = '/raid/mshoeybi/data/bert/wikipedia/wikidump_lines.json'
	assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert wikipedia.PATH != '<wikipedia_path>', \
                                         wikipedia.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)

class roberta(json_dataset):
	"""
	dataset for roberta with arguments configured for convenience

	command line usage: `--train-data roberta`
	"""
	PATH = '/raid/mshoeybi/data/roberta/all_merged/rn_owt_sto_wiki_0.7_aug22/rn_owt_sto_wiki_0.7.json'
	assert_str = "make sure to set PATH at line 27 of data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert roberta.PATH != '<roberta_path>', \
                                         roberta.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(roberta, self).__init__(roberta.PATH, **kwargs)

class BooksCorpus(json_dataset):
        #PATH = '/home/universal-lm-data.cosmos549/datasets/BooksCorpus/books_lines.jsonl'
        PATH = '/raid/mshoeybi/data/bert/BooksCorpus/books_lines.jsonl'
        def __init__(self, **kwargs):
                if not kwargs:
                        kwargs = {}
                kwargs['text_key'] = 'text'
                kwargs['label_key'] = 'path'
                kwargs['loose_json'] = True
                super(BooksCorpus, self).__init__(BooksCorpus.PATH, **kwargs)

class Reddit(json_dataset):
        PATH = '/raid/mshoeybi/data/gpt2/adlr/urls_55M_ftNA_17M_sub_100_115_ftfy.json'
        #PATH='/home/universal-lm-data.cosmos549/datasets/OpenWebText/json_data/urls_55M_ftNA_17M_sub_100_115_ftfy.json'
        #PATH = '/raid/mshoeybi/data/gpt2/skylion007/openwebtext.jsonl'
        def __init__(self, **kwargs):
                if not kwargs:
                        kwargs = {}
                kwargs['text_key'] = 'text'
                kwargs['loose_json'] = True
                super(Reddit, self).__init__(Reddit.PATH, **kwargs)


class RedditAll(json_dataset):
        PATH = '/home/universal-lm-data.cosmos549/datasets/OpenWebText/json_data/reddit_all_ftfy.json'
        #PATH = '/raid/mshoeybi/data/gpt2/skylion007/openwebtext.jsonl'
        def __init__(self, **kwargs):
                if not kwargs:
                        kwargs = {}
                kwargs['text_key'] = 'text'
                kwargs['loose_json'] = True
                super(RedditAll, self).__init__(RedditAll.PATH, **kwargs)


class RedditAllLg200(json_dataset):
        PATH = '/home/universal-lm-data.cosmos549/datasets/OpenWebText/json_data/reddit_all_ftfy_lg200.json'
        #PATH = '/raid/mshoeybi/data/gpt2/skylion007/openwebtext.jsonl'
        def __init__(self, **kwargs):
                if not kwargs:
                        kwargs = {}
                kwargs['text_key'] = 'text'
                kwargs['loose_json'] = True
                super(RedditAllLg200, self).__init__(RedditAllLg200.PATH, **kwargs)



NAMED_CORPORA = {
	'wikipedia': wikipedia,
        'roberta': roberta,
        'BooksCorpus': BooksCorpus,
        'Reddit': Reddit,
        'RedditAll': RedditAll,
        'RedditAllLg200': RedditAllLg200,
}
