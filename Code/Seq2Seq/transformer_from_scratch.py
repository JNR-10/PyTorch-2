"""
Coding the entire "Attention is all you need paper from scratch in PyTorch"
"""

import torch
import torch.nn as nn

# Starting with the most complicated thing, ie. Self Attention
class SelfAttention(nn.Module):
    # Initialize the class with the embedding size and number of heads
    # Embed size is the size of the embedding vector
    # Heads is the number of heads in the self attention (parts we split the embedding into)
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # if not divisible, raise error
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        # Linear layers for Q, K, V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # mask is for padding
        N = query.shape[0] # number of training examples at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # multiply queries and keys to get attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim) -> nqhd
        # keys shape: (N, key_len, heads, head_dim) -> nkhd
        # energy shape: (N, heads, query_len, key_len) -> nhqk
        """
        What is einsum? 
        It is a way to do matrix multiplication in a more general way, if not this we would have to use for loop to do this
        batch matrix multiplication and before that we would have to flatten the matrix and then do the matrix multiplication
        """

        # adding a mask to the energy that will be used to ignore the padding in the case of decoder
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # if the mask is 0, then we set the energy to a very small number, essentially shutting it off

        # apply softmax to the energy to get the attention scores
        # Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k))V
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) # dim=3 because we want to apply softmax on the last dimension
        # attention shape: (N, heads, query_len, key_len)

        # multiply attention with values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        ) # this is the concatenation of the heads

        # out shape: (N, query_len, heads, head_dim)
        # values shape: (N, value_len, heads, head_dim) -> nlhd
        # attention shape: (N, heads, query_len, key_len) -> nhql
        # out shape: (N, query_len, heads, head_dim) -> nqhd
        # fc_out shape: (N, query_len, heads, head_dim) -> nqhd
        # fc_out is the final output of the self attention layer
        # we then pass this to the feed forward network
        
        # pass out to fc_out
        out = self.fc_out(out)
        # out shape: (N, query_len, embed_size)
        return out


# Now we will create the transformer block
class TransformerBlock(nn.Module):
    # forward_expansion is the expansion factor for the feed forward network
    # we send dropout because we will be using it to apply dropout to the self attention and feed forward network
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            # taking embed_size as input and mapping it to the forward_expansion * embed_size as output
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size), # another linear layer to map it back to embed_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask):
        # values, keys, query are the same as in self attention
        # mask is for padding
        attention = self.attention(values, keys, query, mask)
        # attention shape: (N, query_len, embed_size)
        
        x = self.dropout(self.norm1(attention + query)) # skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        # out shape: (N, query_len, embed_size)

        return out


# now comes the encoder
class Encoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length, # this is the maximum length of the input sequence, will be used for positional encoding, handling the case where some inputs are too large
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, 
                                heads, 
                                dropout=dropout, 
                                forward_expansion=forward_expansion,
                ) 
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): # x shape: (N, seq_length)
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # positions shape: (N, seq_length)
        out = self.dropout((self.embedding(x) + self.positional_encoding(positions))) # this is positional encoding part

        for layer in self.layers:
            out = layer(out, out, out, mask) # as values, keys, query are the same as in self attention in the first input
        # out shape: (N, seq_length, embed_size)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self, 
        embed_size, 
        heads, 
        forward_expansion, 
        dropout, 
        device,
    ):
        super(DecoderBlock, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, value, key, src_mask, trg_mask):
        # value, key are the same as in self attention (from encoder)
        # src_mask is for padding (masking encoder outputs)
        # trg_mask is for masking the future tokens (for self attention)
        attention = self.self_attention(x, x, x, trg_mask)
        # attention shape: (N, query_len, embed_size)
        x = self.dropout(self.norm1(attention + x)) # skip connection
        out = self.transformer_block(value, key, x, src_mask)
        # out shape: (N, query_len, embed_size)
        return out


class Decoder(nn.Module):
    def __init__(
        self, 
        trg_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, 
                            heads, 
                            forward_expansion, 
                            dropout, 
                            device,
                ) for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # positions shape: (N, seq_length)
        x = self.dropout((self.embedding(x) + self.positional_encoding(positions))) # this is positional encoding part

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        # x shape: (N, seq_length, embed_size)
        out = self.fc_out(x) # here we will get prediction for the next token
        # out shape: (N, seq_length, trg_vocab_size)
        return out


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx, 
        trg_pad_idx, 
        src_max_length = 100, 
        trg_max_length = 100, 
        embed_size = 256, 
        num_layers = 6, 
        heads = 8, 
        device = 'cuda' if torch.cuda.is_available() else 'cpu', 
        forward_expansion = 4, 
        dropout = 0.1, 
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(
            src_vocab_size, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            src_max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            trg_max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # this will make the triangular mask for the target sequence
        # trg_mask shape: (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        # enc_out shape: (N, src_len, embed_size)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        # out shape: (N, trg_len, trg_vocab_size)
        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device) # not necessarily same size as input

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx, 
        trg_pad_idx, 
    ).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape) # (N, trg_len - 1, trg_vocab_size)
    
    