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
    
class PostionalEmbedding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #creating a matix of shape (seq_len,d_model)
        pe = torch.zros(seq_len,d_model)
        #creating a voctor of shape (seq_len,1)
        position = torch.arange(seq_len,dype = torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        #applying sine and cosine 
        pe[:,0::2] = torch.sin(position*denominator)
        pe[:,1::2] = torch.cos(position*denominator)

        #adding batch dimesion (1,sqe_len,d_model)
        pe = pe.unsqueeze(0) 

        self.register_buffer('pe',pe) # making it in registered and saved.

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).required_grad(False)
        return self.dropout(x)

