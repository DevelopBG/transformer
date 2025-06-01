import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)