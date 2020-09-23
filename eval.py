import torch
import os
from allennlp.data import PyTorchDataLoader as DataLoader
from utils import DSLSharedTaskDataset, Instance
from allennlp.common.params import Params
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialization_dir", type=str, default="weights/")
    parser.add_argument("--model_checkpoint", type=str, default="best.th")
    parser.add_argument("--config", type=str, default="config.json")
    
    args = parser.parse_args()

    # Load dataset
    dataset_reader = DSLSharedTaskDataset()
    test_data = dataset_reader.read('DSL-Task/data/DSLCC-v2.1/gold/A_pt.txt')

    weights_file = os.path.join(args.serialization_dir, args.model_checkpoint)
    config = Params.from_file(args.config)
    config["data_loader"]["shuffle"] = 0 # shuffle disabled for evaluation
    model = Model.load(config, args.serialization_dir, weights_file=weights_file, cuda_device=0)

    # Now that I have the model, I can access the vocab and create the dataloader
    test_data.index_with(model.vocab)
    test_loader = DataLoader(test_data, batch_size=8)

    # Perform inference and get the results
    results = evaluate(model, test_loader, cuda_device=0)
    print(results)
