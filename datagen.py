# Creating preprocessed PyTorch Datagenrators for 
# 1. Encoder Input - Source sequences
# 2. Decoder Input - Target sequences
# 3. Decoder Output - Target sequences shifted by 1


import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm


class TranslationDataset(Dataset):
    def __init__(self, 
                 source_tokens: List[List[int]], 
                 target_tokens: List[List[int]], 
                 max_length: int = 128,
                 pad_token_id: int = 0):
        """
        Dataset for handling translation data.
        
        Args:
            source_tokens: List of tokenized source sentences
            target_tokens: List of tokenized target sentences
            max_length: Maximum sequence length
            pad_token_id: ID of padding token
        """
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
    def __len__(self) -> int:
        return len(self.source_tokens)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src = self.source_tokens[idx]
        tgt = self.target_tokens[idx]
        
        # Truncate if needed
        src = src[:self.max_length]
        tgt = tgt[:self.max_length]
        
        return src, tgt
    

class DataGenerator:
    def __init__(self, 
                 batch_size: int = 64,
                 max_length: int = 128,
                 pad_token_id: int = 0,
                 num_workers: int = 4):
        """
        Initialize the data generator.
        
        Args:
            tokenizer: Trained tokenizer instance
            batch_size: Batch size for training
            max_length: Maximum sequence length
            pad_token_id: ID of padding token
            num_workers: Number of workers for data loading
        """
        # self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.num_workers = num_workers

    def create_dataloaders(self,
                      source_tokens: List[str],
                      target_tokens: List[str],
                      val_source_tokens: Optional[List[str]] = None,
                      val_target_tokens: Optional[List[str]] = None,
                      shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create training and validation dataloaders.
        
        Args:
            source_tokens: List of source language tokens for training
            target_tokens: List of target language tokens for training
            val_source_tokens: Optional list of source language tokens for validation
            val_target_tokens: Optional list of target language tokens for validation
            shuffle: Whether to shuffle the training data
            
        Returns:
            Tuple of training and validation dataloaders
        """
        # Create training dataset
        train_dataset = TranslationDataset(
            source_tokens, target_tokens, 
            self.max_length, self.pad_token_id
        )
        
        # Create training dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        # Create validation dataloader if validation data is provided
        val_loader = None
        if val_source_tokens is not None and val_target_tokens is not None:
            val_dataset = TranslationDataset(
                val_source_tokens, val_target_tokens, 
                self.max_length, self.pad_token_id
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn
            )
        
        return train_loader, val_loader
    
    def collate_fn(self, batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for creating batches with consistent sequence lengths.
        
        Args:
            batch: List of (source, target) token pairs
                
        Returns:
            Tuple of (source, target input, target labels) tensors
            All tensors will have shape [batch_size, max_length]
        """
        # Separate source and target sequences
        source_seqs, target_seqs = zip(*batch)
        
        # Pad sequences
        source_padded = self._pad_sequences(source_seqs)
        target_padded = self._pad_sequences(target_seqs)
        
        # Convert padded sequences to tensors
        source_tensor = torch.LongTensor(source_padded)  # [batch_size, max_length]
        target_tensor = torch.LongTensor(target_padded)  # [batch_size, max_length]
        
        # Create target input and labels with proper padding
        target_input = torch.full((len(batch), self.max_length), self.pad_token_id, dtype=torch.long)
        target_labels = torch.full((len(batch), self.max_length), self.pad_token_id, dtype=torch.long)
        
        # Fill target input (exclude last token) and target labels (exclude first token)
        target_input[:, :-1] = target_tensor[:, :-1]  # Input: [<bos> w1 w2 w3 ... wn-1 pad]
        target_labels[:, :-1] = target_tensor[:, 1:]  # Labels: [w1 w2 w3 ... wn pad pad]
        
        return source_tensor, target_input, target_labels
        
    def _pad_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Pad sequences to the fixed maximum length."""
        padded_seqs = []
        
        for seq in sequences:
            # Truncate if needed
            seq = seq[:self.max_length]
            # Pad to max_length (not just the longest in batch)
            padded = seq + [self.pad_token_id] * (self.max_length - len(seq))
            padded_seqs.append(padded)
            
        return padded_seqs