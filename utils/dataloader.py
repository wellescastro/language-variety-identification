from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, SpacyTokenizer

import io

@DatasetReader.register('classification-dsl')
class DSLSharedTaskDataset(DatasetReader):
    def __init__(self):
        super(DSLSharedTaskDataset, self).__init__(lazy=False)
        self.tokenizer = SpacyTokenizer()
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _read(self, text_path: str) -> Iterable[Instance]:
        with open(text_path, "r") as text_data:
            text_data = text_data.read().splitlines()
            for line in text_data:
                try:
                    text, label = line.strip().split('\t')
                except ValueError:
                    print(line)
                text_field = TextField(self.tokenizer.tokenize(text),
                                        self.token_indexers)
                label_field = LabelField(label)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)
                
    def text_to_instance(self, text: str, label: str = None) -> Instance:
            tokens = self.tokenizer.tokenize(text)
            text_field = TextField(tokens, self.token_indexers)
            fields = {'text': text_field}
            if label:
                fields['label'] = LabelField(label)
            return Instance(fields)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", type=str)
    args = parser.parse_args()

    txt_path = args.text_path

    dataset_reader = DSLSharedTaskDataset()
    instances = dataset_reader.read(txt_path)