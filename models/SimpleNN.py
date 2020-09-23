from typing import Dict, Iterable, List
import torch

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

@Model.register('simple_nn')
class SimpleNN(Model):
    model_state = [
        "validation",
        "train"
    ]
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.hidden_layer = torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.utils.weight_norm(torch.nn.Linear(encoder.get_output_dim(), 128)), 
                                torch.nn.LeakyReLU(inplace=True),
                            )
        self.output_layer = torch.nn.Linear(128, num_labels)

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        x = self.hidden_layer(encoded_text)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        # Shape: (batch_size, num_labels)
        logits = self.output_layer(x)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)

        self.accuracy(logits, label)
        return {'loss': loss, 'probs': probs}
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy_" + self.model_state[self.training]: self.accuracy.get_metric(reset)}