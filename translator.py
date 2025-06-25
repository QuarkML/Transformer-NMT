import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from tqdm import tqdm
from preprocess import preprocess_text
from tokenizer import BilingualTokenizer as Tokenizer
from model import Transformer, TransformerConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import generate_square_subsequent_mask
import torch.nn.functional as F



class TranslationInference:
    def __init__(self, model, tokenizer, device, max_length=128):
        """
        Initialize inference setup
        
        Args:
            model: Trained translation model
            tokenizer: Tokenizer with encode/decode methods for both languages
            device: torch device (cuda/cpu)
            max_length: Maximum sequence length for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
        
    def translate_text(self, source_text):
        """
        Translate a single text input
        
        Args:
            source_text: String of text in source language
            
        Returns:
            String of translated text in target language
        """
        # Preprocess the text
        preprocessed_text, _ = preprocess_text([source_text], [source_text])

        # Raw token encoding
        encoded_src = self.tokenizer.encode_en_raw(preprocessed_text[0])

        # The source tensor which is being passed to the encoder
        src_tensor = torch.tensor(encoded_src, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate translation
        with torch.no_grad():
            # Start with BOS token
            bos_token_id = 2
            eos_token_id = 3
            decoder_input = torch.full(
                (1, 1),
                bos_token_id,
                dtype=torch.long,
                device=self.device
            )
            
            # Generate tokens one by one
            for _ in range(self.max_length - 1):
                # Generate mask for current sequence length
                future_mask = generate_square_subsequent_mask(
                    decoder_input.size(1),
                    device=self.device
                )
                
                # Get model predictions shape: (batch_size, seq_len, vocab_size)
                logits = self.model(src_tensor, decoder_input, future_mask)
                
                # Get next token (use last position only): shape (1, vocab_size)
                next_token_logits = logits[:, -1, :]

                # Get the token with the highest probability
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
                # Append to decoder input
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == eos_token_id:
                    break
                
                # Stop if maximum length is reached
                if decoder_input.size(1) >= self.max_length:
                    break
        
        # Decode the generated tokens
        translated_text = self.tokenizer.decode_hi(decoder_input[0].tolist())
        
        return translated_text