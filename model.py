import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super.__init__()
        self.d_model = d_model # size of embedding vector  
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
        pe = torch.zeros(seq_len,d_model)
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
    

# layer norm 
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6):
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # added

    def forward(self,x):
        mean =  x.mean(dim=-1,keepdim = True)
        std = x.std(dim=-1,keepdim = True)
        return self.alpha*(x-mean)/(std+self.eps) +self.beta
    
# feed forward network----

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)   # w2 and b2

    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
# starting of attention layer--
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int, h:int, dropout:float) ->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):

        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k) # q x k^T

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def forward(self, q,k,v,d,mask):

        query = self.w_q(q) # (batch,seq_len,d_model) --> (batch,seq_len, d_model )
        key = self.w_k(k)
        value = self.w_v(v)

        # transposing -> as we want each head to see full sentence but different embedding( seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value,mask, dropout=self.dropout)
        
        # ( batch_size, h, seq_len, d_k) --> (batch_size, seq_len,h, d_k) --> (batch_size, seq_len, d_model)
        x =  x.transpose(1,2).contiguous().view(x.shapep[0],-1,self.h*self.d_k)

        return self.w_o(x)


        
