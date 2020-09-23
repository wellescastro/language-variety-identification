import torch
import os
import json
from copy import copy
from allennlp.data import PyTorchDataLoader as DataLoader
from utils import DSLSharedTaskDataset, Instance
from allennlp.common.params import Params
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator
from typing import Iterable
from allennlp.data import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from models import SimpleNN, Model
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
# embedding modules
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

from allennlp.training.util import evaluate

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=30, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=30)
    return SimpleNN(vocab, embedder, encoder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialization_dir", type=str, default="weights/")
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    serialization_dir = args.serialization_dir
    with open(args.config, "r") as config_f:
        params = Params(json.loads(config_f.read()))


    # 1. setting up dataset, vocab and dataloaders
    dataset_reader = DSLSharedTaskDataset()

    train_dataset = dataset_reader.read(params["train_data_path"])
    valid_dataset = dataset_reader.read(params["validation_data_path"])

    vocab = build_vocab(train_dataset + valid_dataset)
    train_dataset.index_with(vocab)
    valid_dataset.index_with(vocab)
    data_loader_params = params.pop('data_loader')
    batch_size = data_loader_params['batch_size']
    train_loader = DataLoader.from_params(dataset=train_dataset, params=data_loader_params)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 2. setting up model and training details

    # model = build_model(vocab)
    model = Model.from_params(vocab=vocab, params=params["model"])
    model.cuda()
    trainer = Trainer.from_params(
        model=model, 
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=valid_loader,
        params=params['trainer'],       
    )

    trainer.optimizer.weight_decay = 0.00001
 
    # 3. perform training with early stopping
    print("Starting training")
    trainer.train()
    print("Finished training")
    # save vocabulary for further use
    vocabulary_dir = os.path.join(serialization_dir, 'vocabulary')
    vocab.save_to_files(vocabulary_dir)

    # 5. Verify performance against the test subset
    test_data = dataset_reader.read(params["test_data_path"])
    test_data.index_with(model.vocab)
    test_loader = DataLoader(test_data, batch_size=8)

    results = evaluate(model, test_loader, cuda_device=0)
    print(results)
