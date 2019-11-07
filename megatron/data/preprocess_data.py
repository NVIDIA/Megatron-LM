
import json
import nltk
nltk.download('punkt')

from bert_tokenization import FullTokenizer


def document_generator_provider(input_file):
    with open(input_file, 'r') as ifile:
        for document in ifile:
            data = json.loads(document)
            text = data['text']
            sentences = []
            for line in text.split('\n'):
                if line != '\n':
                    sentences.extend(nltk.tokenize.sent_tokenize(line))
            yield sentences


if __name__ == '__main__':

    print('processing data ...')

    input_file = '/raid/mshoeybi/data/albert/sample/samples_11.json'
    vocab_file = '/raid/mshoeybi/data/albert/bert_vocab/vocab.txt'

    tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    document_generator = document_generator_provider(input_file)
    for sentences in document_generator:
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            print(sentence)
            print(tokens)




