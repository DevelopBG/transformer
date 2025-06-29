import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__()
        self.ds  = ds
        self.tokenizer_scr = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype= torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(['[EOS]'])], dype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(['[PAD]'])], dype = torch.int64)

    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_scr.encoder(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.decoder(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2: end of sentence and start of sentence
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1 : start of sentence

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens<0:
            raise  ValueError("Senctence is too long")
        
        # add sos and eos to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64) 
                
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)

            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # for debugging

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encodr_input' : encoder_input, # (seq_len)
            'decoder_input' : decoder_input, #(seq_len)
            'encoder_mask' : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #( 1,1,seq_len)
            'decoder_mask' : (decoder_input  != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1, seq_len, seq_len)
            'label' : label, # seq_len
            'scr_text' : src_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).type(torch.int)
    return mask == 0
