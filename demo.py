import torch
import os
from allennlp.data import PyTorchDataLoader as DataLoader
from utils import DSLSharedTaskDataset, Instance
from allennlp.common.params import Params
from typing import Iterable
from allennlp.data import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from models import SimpleNN, Model, SentenceClassifierPredictor
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
    parser.add_argument("--text", type=str)
    
    args = parser.parse_args()
    serialization_dir = args.serialization_dir
    model_checkpoint = args.model_checkpoint
    params = Params.from_file(args.config)
    text = args.text
    
    dataset_reader = DSLSharedTaskDataset()

    weights_file = os.path.join(serialization_dir, model_checkpoint)
    params["data_loader"]["shuffle"] = 0 # shuffle disabled for evaluation
    model = Model.load(params, args.serialization_dir, weights_file=weights_file, cuda_device=0)
    vocab = model.vocab

    predictor = SentenceClassifierPredictor(model, dataset_reader)

    output = predictor.predict(text)
    print([(vocab.get_token_from_index(label_id, 'labels'), prob)
        for label_id, prob in enumerate(output['probs'])])