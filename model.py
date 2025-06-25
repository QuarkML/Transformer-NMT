# The transformer Encoder-Decoder Architecture 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

class TransformerConfig:
    """Configuration class to store hyperparameters for the transformer model."""
    
    def __init__(self,
                 src_vocab_size: int = 32000,
                 tgt_vocab_size: int = 32000,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.2,
                 max_seq_length: int = 128,
                 pad_token_id: int = 0,
                 shared_embeddings: bool = False):
        """
        Initialize the configuration.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of feed forward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            pad_token_id: Padding token ID
            shared_embeddings: Whether to share embeddings between encoder and decoder
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.shared_embeddings = shared_embeddings


def scaled_dot_product_attention(query: torch.Tensor,
                                key: torch.Tensor,
                                value: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute scaled dot product attention.
    
    Args:
        query: Query tensor (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor (batch_size, num_heads, seq_len, head_dim)
        value: Value tensor (batch_size, num_heads, seq_len, head_dim)
        mask: Optional mask tensor (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
        
    Returns:
        Output tensor after attention (batch_size, num_heads, seq_len, head_dim)
    """
    # Get dimensions
    head_dim = query.size(-1)
    
    # Compute scaled dot product
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply mask if provided, Issue in masking:
    if mask is not None:

        # Using a smaller value to mask out the scores for better numerical stability.

        # NOTE: We are inverting the masks here, this is because of the way masks are created.
        # We are using True for padding tokens and False for non-padding tokens.
        # So we need to invert the mask to get the correct masking.
        scores = scores.masked_fill(~mask, -1e+30 if scores.dtype == torch.float32 else -1e+4)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute final output
    output = torch.matmul(attention_weights, value)
    
    return output

class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize multi-head attention.
        
        Args:
            config: Model configuration

        The multihead-attention is the core module where we split the tasks between multiple heads
        and then combine them back to the original dimension. The query, key, and value are linearly projected
        to the required dimensions and then split into multiple heads. The scaled dot product attention is applied
        to each head and then combined back to the original dimension.
        """
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = config.d_model # 512
        self.num_heads = config.num_heads # 8
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        # The dimension of each head will be 512 / 8 = 64
        self.head_dim = self.d_model // self.num_heads
        
        # Linear projections

        # These are the projection weights for the query, key, and value
        self.q_proj = nn.Linear(self.d_model, self.d_model) # Query projection
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        
        # The output projection that takes the concatenated heads and projects them back to the original dimension
        self.output_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear Projections and Splitting into multiple heads

        # Each tensor will have the shape of (batch_size, num_heads, seq_len, head_dim)
        # Eg: (32, 8, 256, 64), where 64 is the head_dim

        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot product attention
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Perform the output projection to maintain the original dimension
        output = self.output_proj(attn_output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding module using sine and cosine functions."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize positional encoding.
        
        Args:
            config: Model configuration

        The Sin Cos postional encoding is a clever way to encode the position of the words in the sentence
        with a continuous function for better gradient flow, instead of using a one-hot encoding.

        It works by creating a matrix of shape (max_seq_length, d_model) and then applying a sine function to the even indices
        and a cosine function to the odd indices. It will create continious unique indices for each word in the sentence.
        This matrix is then added to the input embeddings.
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Feed forward module with two linear layers."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize feed forward network.
        
        Args:
            config: Model configuration


        The normal Feed-Forward or Multi-Layer Perceptron layer.
        This layer learns the meaning of words based on the context of the sentence.
        It is in this layer, the transformer will hold memory of relationship between different entities
        different ways of representing the same word in different contexts, etc.
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed forward network.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feed forward network."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize encoder layer.
        
        Args:
            config: Model configuration
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.pad_token_id = config.pad_token_id
        
    def forward(self, 
                x: torch.Tensor,
                src_padding_mask: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            src_mask: Optional source mask for masking future positions
            src_padding_mask: Optional padding mask for masking padding tokens
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
    
        # Self attention block with residual connection and layer norm
        # For the encoder, we will pass the padding mask to the self attention layer
        # The padding mask will be of the shape: (batch_size, 1, 1, seq_len)
        attn_output = self.self_attn(x, x, x, src_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed forward block with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x



class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention and feed forward network."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize decoder layer.
        
        Args:
            config: Model configuration
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        
    def forward(self,
               x: torch.Tensor,
               memory: torch.Tensor,
               combined_mask: torch.Tensor,
               memory_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            memory: Output from encoder (batch_size, src_seq_len, d_model)
            tgt_padding_mask: Target padding mask for masking padding tokens
            memory_padding_mask: Memory padding mask for masking padding tokens in encoder output
            future_mask: Mask for masking future positions
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """

        # Self attention block with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, combined_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross attention block with residual connection and layer norm
        attn_output = self.cross_attn(x, memory, memory, memory_padding_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed forward block with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Encoder(nn.Module):
    """Transformer encoder with multiple encoder layers."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize encoder.
        
        Args:
            config: Model configuration
        """
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config)
        
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.pad_token_id = config.pad_token_id
        
    def forward(self,
               src: torch.Tensor,
               src_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of encoder.
        
        Args:
            src: Source tensor (batch_size, src_seq_len)
            src_mask: Source mask for masking padding tokens
            
        Returns:
            Encoder output tensor (batch_size, src_seq_len, d_model)
        """
        
        # Embed tokens and add positional encoding
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, src_padding_mask)
        
        x = self.norm(x)
        return x
    

class Decoder(nn.Module):
    """Transformer decoder with multiple decoder layers."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize decoder.
        
        Args:
            config: Model configuration
        """
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config)
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.pad_token_id = config.pad_token_id
        
    def forward(self,
               tgt: torch.Tensor,
               memory: torch.Tensor,
               tgt_padding_mask: torch.Tensor,
               future_mask: Optional[torch.Tensor],
               memory_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of decoder.
        
        Args:
            tgt: Target tensor (batch_size, tgt_seq_len)
            memory: Output from encoder (batch_size, src_seq_len, d_model)
            tgt_mask: Target mask for masking future positions
            memory_mask: Memory mask for masking padding tokens in encoder output
            
        Returns:
            Decoder output tensor (batch_size, tgt_seq_len, d_model)
        """

        if future_mask is not None:
            combined_mask = self._combine_mask(tgt_padding_mask, future_mask)
        else:
            combined_mask = tgt_padding_mask

        # Embed tokens and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, memory, combined_mask, memory_padding_mask)
        
        x = self.norm(x)
        return x

    def _combine_mask(self, tgt_padding_mask: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
        """
        Combine padding and future masks for decoder self-attention.

        This will help us to only pass a single mask to the decoder self-attention layer reducing
        the overhead of passing multiple masks.
        
        Args:
            tgt_padding_mask: [batch_size, 1, 1, seq_len]
            future_mask: [seq_len, seq_len]
            
        Returns:
            Combined mask [batch_size, 1, seq_len, seq_len]
        """
        batch_size = tgt_padding_mask.size(0)
        seq_len = tgt_padding_mask.size(-1)
        
        padding_mask = tgt_padding_mask.squeeze(2) # [batch_size, 1, seq_len]
        padding_mask = padding_mask.unsqueeze(-1)  # [batch_size, 1, 1, seq_len]
        padding_mask = padding_mask.expand(-1, -1, seq_len, -1)  # [batch_size, 1, seq_len, seq_len]
        
        # 2. Prepare future mask with proper broadcasting dimensions
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        future_mask = future_mask.expand(batch_size, 1, -1, -1)  # [batch_size, 1, seq_len, seq_len]
        
        # 3. Combine masks using logical AND
        # Both masks: False means "mask this position", True means "keep this position"
        combined_mask = padding_mask & ~future_mask
        
        return combined_mask



class Transformer(nn.Module):
    """Transformer model with encoder and decoder."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer model.
        
        Args:
            config: Model configuration
        """
        super(Transformer, self).__init__()
        
        self.config = config
        
        # Create embedding layers or use shared embeddings

        # Shared embeddings are useful when the source and target vocabularies are of the same size
        # This will reduce the number of parameters in the model, thus reducing the memory footprint
        if config.shared_embeddings:
            assert config.src_vocab_size == config.tgt_vocab_size, "Vocab sizes must match for shared embeddings"
            self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx=config.pad_token_id)
            self.decoder_embedding = self.encoder_embedding
        else:
            self.encoder_embedding = None
            self.decoder_embedding = None
        
        # Create encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        if config.shared_embeddings:
            self.encoder.embedding = self.encoder_embedding
            self.decoder.embedding = self.decoder_embedding
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)
        
        # Initialize parameters
        self._reset_parameters()
        
    def forward(self,
               src: torch.Tensor,
               tgt: torch.Tensor,
               future_mask: Optional[torch.Tensor] = None,
               ) -> torch.Tensor:
        """
        Forward pass of transformer model.
        
        Args:
            src: Source tensor (batch_size, src_seq_len)
            tgt: Target tensor (batch_size, tgt_seq_len)
            src_mask: Source mask for masking padding tokens
            tgt_mask: Target mask for masking future positions
            memory_mask: Memory mask for masking padding tokens in encoder output
            
        Returns:
            Output logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Padding tokens are masked out in the source and targer sequences
        src_padding_mask = self._prepare_masks(src)
        tgt_padding_mask = self._prepare_masks(tgt)
        # Encode source: Encoder receives the source sequence and returns the output
        memory = self.encoder(
            src = src, 
            src_padding_mask = src_padding_mask
            )
        
        # Decode target: Decoder receives the target sequence, encoder output, and the future_mask
        output = self.decoder(
            tgt = tgt, 
            memory = memory, 
            tgt_padding_mask = tgt_padding_mask, 
            future_mask = future_mask,
            memory_padding_mask = src_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def _prepare_masks(self, seq_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masks for source and target sequences.
        
        Args:
            seq_batch: Source or target Batch tensor
            
        Returns:
            Padding mask tensor for source and target
        """
        # Creating the source padding mask, must be of the shape: (batch_size, 1, 1, seq_len)
        # This produces a mask where the padding tokens are set to False and the rest are set to True
        mask = (seq_batch != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        return mask
    
    def _reset_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



def generate_square_subsequent_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate square mask for future positions in decoder self-attention.
    
    Args:
        size: Sequence length
        device: Device to create mask on (CPU/GPU)
        
    Returns:
        Square mask tensor of shape [size, size] where:
        - False values indicate positions to attend to
        - True values indicate positions to mask (future positions)
    
    Example:
        For size=4, generates mask:
        [[False,  True,  True,  True],
         [False, False,  True,  True],
         [False, False, False,  True],
         [False, False, False, False]]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.bool()
    if device:
        mask = mask.to(device)
    return mask


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Simple Inference / Demonstration
# ==============================================================================

if __name__ == '__main__':
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define special tokens and build vocabulary
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    
    # Example sentences
    src_sentence_text = "cat sat on the mat"
    # For this demo, target is same as source (simple copy task)
    # In a real translation task, tgt_sentence_text would be the translation
    tgt_sentence_text = "cat sat on the mat" 

    # Build vocabulary using simple split() tokenization
    all_sentences_for_vocab = [src_sentence_text, tgt_sentence_text]
    special_tokens_list = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    
    word_to_idx = {token: i for i, token in enumerate(special_tokens_list)}
    current_idx = len(special_tokens_list)

    for sentence in all_sentences_for_vocab:
        for word in sentence.split(): # Tokenize by splitting
            if word not in word_to_idx:
                word_to_idx[word] = current_idx
                current_idx += 1
    
    idx_to_word = {i: token for token, i in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    PAD_ID = word_to_idx[PAD_TOKEN]

    print("\n--- Vocabulary ---")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Word to ID mapping: {word_to_idx}")
    print(f"PAD ID: {PAD_ID}")

    # 3. Configure the Transformer
    # Using small dimensions for quick demonstration
    config = TransformerConfig(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=32,                 # Dimension of model embeddings
        num_heads=4,                # Number of attention heads
        num_encoder_layers=2,       # Number of encoder layers
        num_decoder_layers=2,       # Number of decoder layers
        d_ff=64,                    # Dimension of feed forward network
        dropout=0.0,                # Dropout rate (set to 0 for deterministic demo)
        max_seq_length=15,          # Maximum sequence length (must be >= actual seq lengths)
        pad_token_id=PAD_ID,
        shared_embeddings=True      # Share embeddings since src/tgt vocabs are the same here
    )
    print("\n--- Model Configuration ---")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")

    # 4. Prepare data
    # Source sentence processing: tokens + EOS + padding
    src_token_list = src_sentence_text.split()
    src_ids_temp = [word_to_idx[token] for token in src_token_list]
    src_ids_temp.append(word_to_idx[EOS_TOKEN]) # Add EOS to source
    
    # Pad to max_seq_length
    padding_needed_src = config.max_seq_length - len(src_ids_temp)
    if padding_needed_src < 0:
        raise ValueError(f"Source sentence too long for max_seq_length={config.max_seq_length}")
    src_padded_ids = src_ids_temp + [PAD_ID] * padding_needed_src
    src_tensor = torch.tensor([src_padded_ids], dtype=torch.long, device=device)

    # Target sentence processing (for decoder input): BOS + tokens + padding
    tgt_token_list = tgt_sentence_text.split()
    tgt_ids_input_temp = [word_to_idx[BOS_TOKEN]] # Start with BOS
    tgt_ids_input_temp.extend([word_to_idx[token] for token in tgt_token_list])
    
    padding_needed_tgt = config.max_seq_length - len(tgt_ids_input_temp)
    if padding_needed_tgt < 0:
        raise ValueError(f"Target sentence too long for max_seq_length={config.max_seq_length}")
    tgt_padded_ids_input = tgt_ids_input_temp + [PAD_ID] * padding_needed_tgt
    tgt_tensor_input = torch.tensor([tgt_padded_ids_input], dtype=torch.long, device=device)

    # Expected output sequence (for conceptual comparison): tokens + EOS + padding
    # This is what the model *should* predict at each step.
    expected_output_ids_temp = [word_to_idx[token] for token in tgt_token_list]
    expected_output_ids_temp.append(word_to_idx[EOS_TOKEN]) # Ends with EOS
    # (No explicit padding needed here for `expected_output_words` list, just for tensor comparison if any)


    # 5. Initialize model
    model = Transformer(config).to(device)
    model.eval() # Set to evaluation mode (e.g., disables dropout)

    # 6. Prepare masks
    # Padding masks for source and target are created internally by model.forward()
    # Future mask (causal mask) for decoder self-attention:
    tgt_seq_len = tgt_tensor_input.size(1) # This will be config.max_seq_length
    future_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)

    # 7. Perform a forward pass
    # This simulates a single step of "teacher-forced" generation or a forward pass during training.
    # For true auto-regressive inference, you'd generate tokens one by one in a loop.
    print("\n--- Input Tensors ---")
    print(f"Source input tensor shape: {src_tensor.shape}")
    print(f"Source input tensor: {src_tensor}")
    print(f"Target input tensor (decoder input) shape: {tgt_tensor_input.shape}")
    print(f"Target input tensor (decoder input): {tgt_tensor_input}")
    # print(f"Future mask for target (sample, True means mask): \n{future_mask}")


    with torch.no_grad(): # No need to compute gradients for this demonstration
        logits = model(src=src_tensor, tgt=tgt_tensor_input, future_mask=future_mask)

    # 8. Interpret output
    print("\n--- Output ---")
    print(f"Logits shape: {logits.shape}") # Expected: (batch_size, tgt_seq_len, tgt_vocab_size)

    # Get predicted token IDs using greedy decoding (taking the argmax)
    predicted_ids = torch.argmax(logits, dim=-1) # Shape: (batch_size, tgt_seq_len)
    print(f"Predicted token IDs tensor (first batch): {predicted_ids[0]}")

    # Convert predicted IDs back to words
    predicted_words_list = [idx_to_word[idx.item()] for idx in predicted_ids[0]]
    
    print("\n--- Tokenized Inputs and Predicted Output ---")
    source_words_with_eos = src_token_list + [EOS_TOKEN]
    print(f"Source input: '{' '.join(source_words_with_eos)}'")
    print(f"   Padded IDs: {src_tensor[0].tolist()}")
    
    target_input_words = [BOS_TOKEN] + tgt_token_list
    print(f"Target input (decoder): '{' '.join(target_input_words)}'")
    print(f"   Padded IDs: {tgt_tensor_input[0].tolist()}")
    
    expected_output_words_list = tgt_token_list + [EOS_TOKEN]
    print(f"Expected output sequence: '{' '.join(expected_output_words_list)}'")
    # (The predicted sequence will include padding up to max_seq_length)
    
    print(f"Predicted output sequence (greedy): '{' '.join(predicted_words_list)}'")
    
    print("\nNote: The model is untrained, so the predicted output will likely be random.")
    print("This demonstration shows the data flow and shapes through the Transformer model.")

    # Count parameters
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters in the model: {num_params:,}")

    print("\n--- For actual auto-regressive generation (not implemented here): ---")
    print("1. Encode `src_tensor` once to get `memory` from the encoder.")
    print("2. Initialize `tgt_input` with `BOS_TOKEN` ID.")
    print("3. Loop for `max_seq_length` steps:")
    print("   a. Pass current `tgt_input` and `memory` to the decoder.")
    print("   b. Get logits for the *last* token position.")
    print("   c. Select the next token ID (e.g., argmax or sampling).")
    print("   d. If token is `EOS_TOKEN` or max length reached, stop.")
    print("   e. Append the new token ID to `tgt_input` and repeat.")