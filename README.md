# TransformerNMT

A PyTorch-based Neural Machine Translation (NMT) system using the Transformer architecture for English-Hindi translation. This repository includes scripts for training, evaluation, data preprocessing, tokenization, and inference.

---

## Features

- Transformer Encoder-Decoder architecture implemented from scratch in PyTorch
- Distributed and mixed-precision training support
- Custom BPE tokenization for English and Hindi using the `tokenizers` library
- Data preprocessing and cleaning utilities
- BLEU evaluation with [sacrebleu](https://github.com/mjpost/sacrebleu)
- Inference and interactive translation scripts
- Model checkpointing and conversion utilities

---

## Requirements

Install the following Python packages:

- torch (PyTorch, with CUDA if using GPU)
- tokenizers
- tqdm
- numpy
- sacrebleu
- tensorboard
- matplotlib
- scipy

You can install all requirements with:

```sh
pip install torch tokenizers tqdm numpy sacrebleu tensorboard matplotlib scipy
```

> **Note:** For distributed/mixed-precision training, ensure you have a compatible CUDA setup and the correct PyTorch version.

---

## Usage

### 1. Data Preparation

- Place your parallel English-Hindi data in the `../Data/parallel-n/` directory as required by the scripts.
- Train or load BPE tokenizers for both languages using [`tokenizer.py`](tokenizer.py).

### 2. Training

To train the model with distributed and mixed-precision training (example for 2 GPUs):

```sh
bash train.sh
```

Or manually:

```sh
torchrun --nproc_per_node=2 --master_port=29500 train.py --distributed --fp16 --shared_embeddings ...
```

See [`train.sh`](train.sh) for all hyperparameters.

### 3. Evaluation

Evaluate BLEU score on a test set:

```sh
python test.py --source path/to/test.en --target path/to/test.hi --output path/to/output.trans
```

### 4. Inference

Interactive translation:

```sh
python infer.py
```

Follow the prompt to enter sentences for translation.

---

## File Overview

- [`train.py`](train.py): Main training script
- [`train.sh`](train.sh): Example shell script for distributed training
- [`test.py`](test.py): BLEU evaluation script
- [`infer.py`](infer.py): Interactive translation
- [`tokenizer.py`](tokenizer.py): BPE tokenizer training and loading
- [`preprocess.py`](preprocess.py): Data cleaning and preprocessing
- [`datagen.py`](datagen.py): PyTorch dataset and dataloader utilities
- [`model.py`](model.py): Transformer model implementation
- [`evaluate.py`](evaluate.py): BLEU evaluation utilities
- [`conversion.py`](conversion.py): Convert DDP checkpoints to single-GPU format

---

## Notes

- Adjust paths to your data and tokenizer files as needed.
- For best performance, use a machine with multiple GPUs and sufficient memory.
- Checkpoints and logs are saved in the `models/` and `runs/` directories by default.

---

## License

MIT License

---

## Citation

If you use this codebase, please cite the original Transformer paper:

> Vaswani et al., "Attention is All You Need", NeurIPS 2017.