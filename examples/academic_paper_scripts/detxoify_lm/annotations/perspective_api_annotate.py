import json
import time
from typing import Dict, Optional, List

import joblib
from googleapiclient import discovery
from googleapiclient.errors import HttpError

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--out-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--total', type=int, default=-1,
                    help='Total number of data')
parser.add_argument('--workers', type=int, default=1,
                   help='Number of worker processes to launch')


class PerspectiveApiScorer:
    """
    This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.
    """

    DEFAULT_ATTRIBUTES = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']

    def __init__(self):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        api_key = ''
        self._service = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_scores(self, input_text: str, requested_attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = requested_attributes if requested_attributes else PerspectiveApiScorer.DEFAULT_ATTRIBUTES

        analyze_request = {
            'comment': {'text': input_text},
            'requestedAttributes': {attribute.upper(): {} for attribute in requested_attributes},
            'spanAnnotations': False,
            'languages': ['en'],
        }

        response = None
        while not response:
            try:
                response = self._service.comments().analyze(body=analyze_request).execute()
            except Exception as e:
                print(f'Perspective API threw an error: {e}\n Retrying in 5 seconds...')
                print(input_text)
                time.sleep(1)

        return {attribute: response['attributeScores'][attribute.upper()]['summaryScore']['value'] for attribute in
                requested_attributes}


def test():
    scorer = PerspectiveApiScorer()
    for i in range(1):
        print(scorer.get_scores("toxic behaviors and nasty negro"))


def split_lines(lines, split):
    tot = len(lines)
    each = tot // split
    return [lines[i:i+each] for i in range(0, tot, each)]

from joblib import Parallel, delayed

scorer = PerspectiveApiScorer()

def get_score(line):
    data = json.loads(line)
    text = data['text']
    text = text.replace("<|endoftext|>", "")
    data['text'] = text
    if not text.strip():
        data['score'] = None
        return json.dumps(data)

    encoded_text = text.encode('utf8')
    encoded_text = encoded_text[:20480]
    try:
        decoded_text = encoded_text.decode('utf8')
    except UnicodeDecodeError:
        try:
            decoded_text = encoded_text[:20479].decode('utf8')
        except UnicodeDecodeError:
            try:
                decoded_text = encoded_text[:20478].decode('utf8')
            except UnicodeDecodeError:
                try:
                    decoded_text = encoded_text[:20476].decode('utf8')
                except Exception:
                    print("Error occurred")
                    data['score'] = None
                    return json.dumps(data)
    data['score'] = scorer.get_scores(decoded_text)
    return json.dumps(data)


def get_scores(lines):
    scorer = PerspectiveApiScorer()
    all_data = []
    for i, line in enumerate(tqdm(lines)):
        data = json.loads(line)
        text = data['text']
        if not text.strip():
            data['score'] = None
            all_data.append(json.dumps(data))
            continue
        encoded_text = text.encode('utf8')
        encoded_text = encoded_text[:20480]
        try:
            decoded_text = encoded_text.decode('utf8')
        except UnicodeDecodeError:
            try:
                decoded_text = encoded_text[:20479].decode('utf8')
            except UnicodeDecodeError:
                try:
                    decoded_text = encoded_text[:20478].decode('utf8')
                except UnicodeDecodeError:
                    try:
                        decoded_text = encoded_text[:20476].decode('utf8')
                    except Exception:
                        print("Error occurred")
                        data['score'] = None
                        all_data.append(json.dumps(data))
                        continue
        data['score'] = scorer.get_scores(decoded_text)
        all_data.append(json.dumps(data))
    return all_data

def get_annotated_datasets(lines, threads=10):
    sub_lines = lines
    splitted_lines = split_lines(sub_lines, threads)
    print(len(sub_lines))
    final = Parallel(n_jobs=threads)(delayed(get_score)(l) for l in splitted_lines)
    import itertools
    finals = list(itertools.chain.from_iterable(final))
    return finals


def main():
    args = parser.parse_args()

    path = args.data_path
    out = args.out_path if args.out_path else path + '-annotated.jsonl'
    print(out)

    fin = open(path, 'r', encoding='utf-8')
    import multiprocessing
    pool = multiprocessing.Pool(args.workers)
    annotated = pool.imap(get_score, fin, 25)
    with open(out, "w") as f:
        if args.total > 0:
            for x in tqdm(annotated, total=args.total):
                f.write(x + '\n')
        else:
            for x in tqdm(annotated):
                f.write(x + '\n')


if __name__ == '__main__':
    main()

