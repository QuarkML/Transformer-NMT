import os
import time
import math
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

# Import your model and data generator
from model import Transformer, TransformerConfig, generate_square_subsequent_mask
from tokenizer import BilingualTokenizer as Tokenizer
from datagen import DataGenerator
from preprocess import preprocess_text, split_sequences
import io
import random


os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss with KL divergence.

    Label smoothing is a regularization technique used to prevent the model from 
    predicting with overely high confidence. Instead of using hard targets (i.e., 0s and 1s),
    we use a soft distribution for the targets. This helps to prevent overfitting and
    improve the generalization of the model.

    Eg: instead of using one-hot encoded targets, we use a distribution like [0.1, 0.9]

    KL divergence is used to measure the difference between two distributions. In this
    case, we compare the predicted distribution with the smoothed target distribution.


    """
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            pred: Prediction logits of shape (batch_size, seq_len, vocab_size)
            target: Target indices of shape (batch_size, seq_len)
            
        Returns:
            Smoothed loss value
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create smoothed labels
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            
            # Fill in the correct indices with higher confidence
            true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
            
            # Create mask for ignored indices
            # This ensures that we are not computing loss for padding tokens
            mask = (target == self.ignore_index).unsqueeze(-1)
            true_dist.masked_fill_(mask, 0.0)
        
        # Compute KL divergence
        loss = -torch.sum(true_dist * pred, dim=-1)
        
        # Apply mask for padding tokens
        valid_mask = (target != self.ignore_index).float()
        loss = (loss * valid_mask).sum() / valid_mask.sum()
        
        return loss
    

class Trainer:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


        # Distributed training initialization
        self.setup_distributed()

        # Setup CUDA
        self.device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Create model, optimizer, and loss function
        self.setup_model()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler(enabled=args.fp16)

        # Setup tensorboard
        if self.args.local_rank == 0:  # Only setup on main process
            self.tb_writer = SummaryWriter(log_dir=args.log_dir)

        self.tokenizer = Tokenizer(vocab_size=args.src_vocab_size)
        self.tokenizer_loaded = self.tokenizer.load_tokenizers('../Data/bi_tokenizers_32k')

        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        #self.best_bleu = 0.0

    def setup_distributed(self):
        """Setup distributed training if enabled."""
        self.world_size = 1
        self.is_distributed = self.args.distributed
        
        if self.is_distributed:
            # Initialize process group

            # Setting up NCCL for distributed GPU training
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.local_rank = dist.get_rank()
            self.args.local_rank = self.local_rank
        else:
            self.local_rank = 0


    def setup_model(self):
        """Setup model, optimizer, and loss function."""
        # Create model
        config = TransformerConfig(
            src_vocab_size=self.args.src_vocab_size,
            tgt_vocab_size=self.args.tgt_vocab_size,
            d_model=self.args.d_model,
            num_heads=self.args.num_heads,
            num_encoder_layers=self.args.num_encoder_layers,
            num_decoder_layers=self.args.num_decoder_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            max_seq_length=self.args.max_seq_length,
            pad_token_id=self.args.pad_token_id,
            shared_embeddings=self.args.shared_embeddings
        )
        
        self.model = Transformer(config)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=False
            )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2), # regularization parameters
            eps=self.args.epsilon, # regularization parameters
            weight_decay=self.args.weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.learning_rate,
            total_steps=self.args.total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Create loss function with label smoothing, custom loss function
        self.criterion = LabelSmoothingLoss(
            smoothing=self.args.label_smoothing,
            ignore_index=self.args.pad_token_id
        )

    def load_data(self, source_tokens, target_tokens, val_source_tokens, val_target_tokens):
        """
        Load data using the DataGenerator.
        
        Args:
            source_tokens: Tokenized source sequences
            target_tokens: Tokenized target sequences
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create data generator
        data_generator = DataGenerator(
            batch_size=self.args.batch_size,
            max_length=self.args.max_seq_length,
            pad_token_id=self.args.pad_token_id,
            num_workers=self.args.num_workers
        )
        
        # Create data loaders
        train_loader, val_loader = data_generator.create_dataloaders(
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            val_source_tokens=val_source_tokens,
            val_target_tokens=val_target_tokens,
            shuffle=not self.is_distributed  # We'll use DistributedSampler if distributed
        )
        
        # Add distributed sampler if needed

        # The distributed training works by splitting the dataset across multiple processes (GPUs)
        # Then the each GPU's creates it own copy of the model and the subset of the data,
        # Then the gradients are computed on each of the subset in each GPU, and the gradients are averaged across all the GPUs
        # to update the model weights. This is done using the DistributedSampler which splits the dataset across the GPUs.
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=self.world_size,
                rank=self.local_rank
            )
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=data_generator.collate_fn
            )
            
            if val_loader is not None:
                val_sampler = DistributedSampler(
                    val_loader.dataset,
                    num_replicas=self.world_size,
                    rank=self.local_rank
                )
                val_loader = DataLoader(
                    val_loader.dataset,
                    batch_size=self.args.batch_size,
                    sampler=val_sampler,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    collate_fn=data_generator.collate_fn
                )
        
        return train_loader, val_loader
    

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        # Set sampler epoch if distributed
        if self.is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        tk = tqdm(train_loader, desc=f"Epoch {epoch}", disable=self.local_rank != 0)
        
        for batch_idx, (src, tgt_input, tgt_labels) in enumerate(tk):
            # Move tensors to device
            src = src.to(self.device) # Eg: src = torch.tensor([[1, 2, 3], [4, 5, 6]])
            tgt_input = tgt_input.to(self.device) # Eg: tgt_input = torch.tensor([[7, 8, 9], [10, 11, 12]])
            tgt_labels = tgt_labels.to(self.device) # Eg: tgt_labels = torch.tensor([[8, 9, 10], [11, 12, 13]])
            

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with autocast(enabled=self.args.fp16, device_type='cuda'):
                # Generate the required masks

                # The source mask is already being generated inside the Encoder itself,
                # So we don't need to generate it here, however if we need any custom masking,
                # Then we can pass the src_mask which is then combined with the source mask inside the Encoder
                # to create the final mask.

                # The target mask is generated using the generate_square_subsequent_mask function
                # This will be used for masking out the future tokens in the target sequence
                future_mask = generate_square_subsequent_mask(self.args.max_seq_length, device=self.device)               
                
                # Forward pass
                logits = self.model(src, tgt_input, future_mask)
    
                
                # Compute loss
                loss = self.criterion(logits, tgt_labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward() # Mixed precision scaling
            
            # Apply gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Update parameters with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update learning rate
            self.scheduler.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Update progress bar
            tk.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to tensorboard
            if self.local_rank == 0 and batch_idx % self.args.log_interval == 0:
                self.tb_writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.tb_writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                

            if batch_idx % 100 == 0 and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            self.global_step += 1

            del logits, loss
        
        # Compute average loss
        avg_loss = epoch_loss / len(train_loader)
        
        # Log epoch summary
        if self.local_rank == 0:
            self.tb_writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            
        return avg_loss

        # Modified validate method to use the improved decoder
    def validate(self, val_loader, epoch):
        """
        Validate the model with improved text decoding
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        # Create markdown table header with proper encoding
        translation_log = "| Source | Target | Predicted |\n|---------|---------|------------|\n"
        
        with torch.no_grad():
            for batch_idx, (src, tgt_input, tgt_labels) in enumerate(tqdm(val_loader, desc="Validating", disable=self.local_rank != 0)):
                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_labels = tgt_labels.to(self.device)
                
                future_mask = generate_square_subsequent_mask(self.args.max_seq_length, device=self.device)
                logits = self.model(src, tgt_input, future_mask)

                # Note: the logits of the model will have the shape [batch_size, seq_len, vocab_size]
                # This means for each token in the sequence of batches, the model produce a 
                # probability distribution tensor, which we can use to compute the loss parallely
                # with the target labels.

                loss = self.criterion(logits, tgt_labels)
                
                val_loss += loss.item()
                num_batches += 1
                

                # Generating translation from validation set
                if batch_idx % 50 == 0 and self.local_rank == 0:
                    curr_batch_size = src.size(0)
                    num_samples = min(1, curr_batch_size)
                    sample_indices = random.sample(range(curr_batch_size), num_samples)
                    
                    for idx in sample_indices:
                        try:
                            # Generate translation
                            src_sequence = src[idx:idx+1]
                            generated = self.generate_translation(src_sequence, max_length=self.args.max_seq_length)

                            src_text = self.tokenizer_loaded.decode_en(src[idx].tolist())
                            tgt_text = self.tokenizer_loaded.decode_hi(tgt_labels[idx].tolist())
                            pred_text = self.tokenizer_loaded.decode_hi(generated[0].tolist())
                            
                            # Log to console for verification
                            print("\nSample Translation:")
                            print(f"Source:    {src_text}")
                            print(f"Target:    {tgt_text}")
                            print(f"Predicted: {pred_text}")
                            print("-" * 50)
                            
                            # Add to markdown table with proper encoding
                            translation_log += f"| {src_text} | {tgt_text} | {pred_text} |\n"
                            
                            # Log to TensorBoard with proper encoding
                            self.tb_writer.add_text(
                                f'translations/batch_{batch_idx}/sample_{idx}',
                                f"""
                                **Source:** {src_text}
                                **Target:** {tgt_text}
                                **Predicted:** {pred_text}
                                """,
                                epoch
                            )
                            
                        except Exception as e:
                            print(f"Error in translation logging: {e}")
                            continue
                
                if batch_idx % 100 == 0 and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Log final translation table
        if self.local_rank == 0:
            self.tb_writer.add_text('translations/epoch_summary', translation_log, epoch)
        
        avg_val_loss = val_loss / num_batches
        if self.local_rank == 0:
            self.tb_writer.add_scalar('val/loss', avg_val_loss, epoch)
        
        return avg_val_loss

    def generate_translation(self, src, max_length=128):
        """
        Generate translation autoregressively.
        
        Args:
            src: Source sequence tensor [1, seq_len]
            max_length: Maximum length of generated sequence
            
        Returns:
            Generated sequence tensor [1, seq_len]
        """
        self.model.eval()
        src = src.to(self.device)
        
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
            for _ in range(max_length - 1):
                # Generate mask for current sequence length
                future_mask = generate_square_subsequent_mask(
                    decoder_input.size(1), 
                    device=self.device
                )
                
                # Get model predictions
                logits = self.model(src, decoder_input, future_mask)
                
                # Get next token (use last position only)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to decoder input
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == eos_token_id:
                    break
                
                # Stop if maximum length is reached
                if decoder_input.size(1) >= max_length:
                    break
        
        return decoder_input
            

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Decoded text string
        """
        # Remove padding tokens
        tokens = token_ids[token_ids != self.args.pad_token_id]
        
        # Convert to list and decode
        token_list = tokens.cpu().tolist()
        
        # Assuming you have tokenizer object available in self.tokenizer
        text = self.tokenizer.decode(token_list, skip_special_tokens=True)
        
        return text

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        if self.local_rank != 0:
            return  # Only save on main process
            
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        # Save checkpoint

        checkpoint_path = "checkpoints"
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, f'checkpoint-epoch-{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'TNMT_Best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")


    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load checkpoint on CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        #bleu_score = checkpoint.get('bleu_score', 0.0)
        
        print(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
        
        return epoch, val_loss
    

    
    def train(self, source_tokens, target_tokens, val_source_tokens, val_target_tokens, is_pre_test):
        """
        Main training loop.
        
        Args:
            source_tokens: Tokenized source sequences
            target_tokens: Tokenized target sequences
        """
        # Create data loaders
        train_loader, val_loader = self.load_data(source_tokens, 
                                                  target_tokens, 
                                                  val_source_tokens, 
                                                  val_target_tokens)
        
        # Load checkpoint if specified

        # Resuming the training run from a checkpoint is useful when you want to continue training
        start_epoch = 0
        if self.args.resume_from:
            start_epoch, self.best_val_loss = self.load_checkpoint(self.args.resume_from)
            start_epoch += 1  # Start from next epoch
        
        # Training loop
        for epoch in range(start_epoch, self.args.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
            
                val_loss = self.validate(val_loader, epoch)
 
                is_best = False
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                    
                # Save checkpoint
                if not is_pre_test:

                    if self.local_rank == 0:
                        self.save_checkpoint(epoch, val_loss, is_best)
                    
                    # Log epoch summary
                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                # Save checkpoint without validation

                if not is_pre_test:
                    if self.local_rank == 0 and epoch % self.args.save_interval == 0:
                        self.save_checkpoint(epoch, float('inf'), 0.0)
                        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        # Final logging
        if self.local_rank == 0:
            self.tb_writer.close()
            print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")




def load_parallel_corpus(en_file: str, hi_file: str) -> Tuple[List[str], List[str]]:
    """
    Efficiently load parallel corpus from English and Hindi text files.
    Returns two lists containing aligned sentences.
    
    Args:
        en_file (str): Path to English corpus file
        hi_file (str): Path to Hindi corpus file
    
    Returns:
        Tuple[List[str], List[str]]: Tuple containing (english_sentences, hindi_sentences)
    """
    # Pre-allocate lists with an initial capacity
    english_sentences = []
    hindi_sentences = []
    
    # Use context managers to ensure proper file handling
    with io.open(en_file, 'r', encoding='utf-8') as en_f, \
         io.open(hi_file, 'r', encoding='utf-8') as hi_f:
        
        # Read files line by line to avoid loading entire file into memory
        for en_line, hi_line in zip(en_f, hi_f):
            # Strip whitespace and add non-empty sentences
            en_sent = en_line.strip()
            hi_sent = hi_line.strip()
            
            if en_sent and hi_sent:  # Only add if both sentences exist
                english_sentences.append(en_sent)
                hindi_sentences.append(hi_sent)
    
    return english_sentences, hindi_sentences


def main():
    parser = argparse.ArgumentParser(description="Neural Machine Translation with Transformer")
    
    # Model parameters
    parser.add_argument("--src_vocab_size", type=int, default=32000, help="Source vocabulary size")
    parser.add_argument("--tgt_vocab_size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Padding token ID")
    parser.add_argument("--shared_embeddings", action="store_true", help="Share embeddings between encoder and decoder")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps for LR scheduler")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="runs/tnmt_training", help="Directory for tensorboard logs")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval")
    parser.add_argument("--translation_interval", type=int, default=500, help="Translation printing interval")
    parser.add_argument("--resume_from", type=str, default="", help="Resume from checkpoint")
    
    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)

    # Data Loading and Preprocessing
    train_en, train_hi = load_parallel_corpus("../Data/parallel-n/english_subset.txt", "../Data/parallel-n/hindi_subset.txt")
    val_en, val_hi = load_parallel_corpus("../Data/parallel-n/dev.en", "../Data/parallel-n/dev.hi")
    train_en, train_hi = preprocess_text(train_en, train_hi)
    val_en, val_hi = preprocess_text(val_en, val_hi)

    # Loading tokenizers
    tokenizer = Tokenizer(vocab_size=args.src_vocab_size)
    tokenizer_loaded = tokenizer.load_tokenizers('../Data/bi_tokenizers_32k')

    # Tokenizing src and tgt
    train_en, train_hi = tokenizer_loaded.encode_(train_en, train_hi, add_special_tokens=True)
    val_en, val_hi = tokenizer_loaded.encode_(val_en, val_hi, add_special_tokens=True)

    # Let's train the model.
    trainer.train(train_en, train_hi, val_en, val_hi, is_pre_test = False)


if __name__ == "__main__":
    # torchrun --nproc_per_node=2 train.py --distributed
    main()