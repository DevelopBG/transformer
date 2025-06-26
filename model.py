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
    
class PositionalEmbedding(nn.Module):
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


    def forward(self, q,k,v,mask):

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
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout:float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

 # Now we will create encoder block
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attantion_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])

    def forward(self,x, scr_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,scr_mask)) 
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x



class Encoder( nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):

        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x,encoder_output,scr_mask,tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,scr_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self,x,encoder_output,scr_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,scr_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.liner = nn.Linear(d_model,vocab_size)
    def forward(self, x):
        return torch.log_softmax(self.liner(x),dim=-1) # log softmax for numerical stability
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEmbedding,tgt_pos:PositionalEmbedding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

         
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self,encoder_output, src_mask, tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output,src_mask, tgt_mask)

    def project(self,x):
        return self.projection_layer(x)
    
# build transformer 

def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int,d_model:int=512,N:int=6, h:int=8,dropout:float=0.1,d_ff:int=2048):
    
    # create embedding layer
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    # create the positional encoding layer
    src_pos = PositionalEmbedding(d_model,src_seq_len)
    tgt_pos = PositionalEmbedding(d_model,tgt_seq_len)

    # creating encoder 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block)
        encoder_blocks.append(encoder_block)
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model,tgt_vocab_size) # Liner layer

    transformer = Transformer(encoder, decoder,src_embed, tgt_embed,src_pos, tgt_pos,projection_layer)

    # initialization of the parameter

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer





