from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
from typing import List, Tuple, Union
from tqdm import tqdm
import os
import torch
import re

class BilingualTokenizer:
    def __init__(self, vocab_size: int = 16000):
        """
        Initialize tokenizers for both languages with proper configuration.
        
        Args:
            vocab_size (int): Size of vocabulary for each tokenizer
        """
        self.vocab_size = vocab_size
        
        # English tokenizer setup
        self.en_tokenizer = Tokenizer(models.BPE())
        self.en_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.en_tokenizer.decoder = decoders.ByteLevel()
        
        # Hindi tokenizer setup with specific configuration for Devanagari
        self.hi_tokenizer = Tokenizer(models.BPE())
        self.hi_tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),  # Unicode normalization
            normalizers.Strip()   # Remove leading/trailing whitespace
        ])
        self.hi_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Metaspace()
        ])
        self.hi_tokenizer.decoder = decoders.Metaspace()

    def train_tokenizers(self, 
                        en_sentences: List[str], 
                        hi_sentences: List[str],
                        save_dir: str = "tokenizers") -> None:
        """
        Train separate tokenizers for English and Hindi.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare training files
        print("Preparing training files...")
        en_path = os.path.join(save_dir, "train_en.txt")
        hi_path = os.path.join(save_dir, "train_hi.txt")
        
        with open(en_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(en_sentences))
        
        with open(hi_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hi_sentences))
        
        # Configure trainers
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        
        # Train English tokenizer
        print("\nTraining English tokenizer...")
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=special_tokens
        )
        self.en_tokenizer.train([en_path], trainer)
        
        # Train Hindi tokenizer
        print("\nTraining Hindi tokenizer...")
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=special_tokens
        )
        self.hi_tokenizer.train([hi_path], trainer)
        
        # Save tokenizers
        print("\nSaving tokenizers...")
        self.en_tokenizer.save(os.path.join(save_dir, "en_tokenizer.json"))
        self.hi_tokenizer.save(os.path.join(save_dir, "hi_tokenizer.json"))
        
        # Clean up
        os.remove(en_path)
        os.remove(hi_path)
        
        print(f"\nTokenizers saved to {save_dir}")

    @classmethod
    def load_tokenizers(cls, save_dir: str) -> 'BilingualTokenizer':
        """
        Load pre-trained tokenizers from directory.
        """
        instance = cls()
        
        instance.en_tokenizer = Tokenizer.from_file(
            os.path.join(save_dir, "en_tokenizer.json")
        )
        
        instance.hi_tokenizer = Tokenizer.from_file(
            os.path.join(save_dir, "hi_tokenizer.json")
        )
        
        return instance
    

    def encode_(self, 
               en_texts: List[str], 
               hi_texts: List[str],
               add_special_tokens: bool = True,
               batch_size: int = 1000) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Encode texts using the trained tokenizers.
        """
        en_encoded = []
        hi_encoded = []
        
        for i in tqdm(range(0, len(en_texts), batch_size), desc="Encoding texts"):
            batch_en = en_texts[i:i + batch_size]
            batch_hi = hi_texts[i:i + batch_size]
            
            # Encode with proper handling of special tokens
            en_encodings = self.en_tokenizer.encode_batch(batch_en)
            hi_encodings = self.hi_tokenizer.encode_batch(batch_hi)
            
            if add_special_tokens:
                en_encoded.extend([[2] + e.ids + [3] for e in en_encodings])  # 2=BOS, 3=EOS
                hi_encoded.extend([[2] + h.ids + [3] for h in hi_encodings])
            else:
                en_encoded.extend([e.ids for e in en_encodings])
                hi_encoded.extend([h.ids for h in hi_encodings])
        
        return en_encoded, hi_encoded
    
    def encode_en_raw(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode English text to token IDs.
        text: string
        """

        encoding = self.en_tokenizer.encode(text)
        if add_special_tokens:
            return [2] + encoding.ids + [3]
        else:
            return encoding.ids
        
    def encode_hi_raw(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode Hindi text to token IDs.
        text: string
        """
        text_batch = [text]
        text_encoded = self.hi_tokenizer.encode_batch(text_batch)
        if add_special_tokens:
            return [2] + text_encoded[0].ids + [3]
        else:
            return text_encoded[0].ids



    def decode_hi(self, tensor_ids: Union[torch.Tensor, List[List[int]]]) -> str:
        """
        Decode Hindi token IDs back to text with proper handling of Devanagari.
        Removes special characters and combines subwords.
        
        Args:
            tensor_ids: Token IDs as either torch.Tensor or List[List[int]]
        
        Returns:
            str: Decoded and cleaned Hindi text
        """
        if isinstance(tensor_ids, torch.Tensor):
            token_ids = tensor_ids.tolist()
        else:
            token_ids = tensor_ids
            
        if not isinstance(token_ids[0], list):
            token_ids = [token_ids]
            
        decoded_texts = []
        
        for sequence in token_ids:
            # Filter out special tokens
            filtered_ids = [id for id in sequence if id not in [0, 1, 2, 3]]
            
            # Decode with proper handling of Hindi characters
            decoded = self.hi_tokenizer.decode(filtered_ids)
            
            # Clean up the decoded text:
            # 1. Replace "▁" with space
            # 2. Remove extra spaces
            # 3. Strip leading/trailing spaces
            cleaned_text = (decoded
                        .replace("▁", "")  # Replace special char with space
                        .replace("  ", " ")  # Remove double spaces
                        .strip())           # Remove leading/trailing spaces
            
            decoded_texts.append(cleaned_text)
        
        return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts

    def decode_en(self, tensor_ids: Union[torch.Tensor, List[List[int]]]) -> str:
        """
        Decode English token IDs back to text.
        """
        if isinstance(tensor_ids, torch.Tensor):
            token_ids = tensor_ids.tolist()
        else:
            token_ids = tensor_ids
            
        if not isinstance(token_ids[0], list):
            token_ids = [token_ids]
            
        decoded_texts = []
        
        for sequence in token_ids:
            filtered_ids = [id for id in sequence if id not in [0, 1, 2, 3]]
            decoded = self.en_tokenizer.decode(filtered_ids)
            decoded_texts.append(decoded)
        
        return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
    