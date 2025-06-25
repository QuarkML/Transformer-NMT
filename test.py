from evaluate import evaluate_bleu
from translator import TranslationInference
import torch
from model import Transformer, TransformerConfig
from tokenizer import BilingualTokenizer as Tokenizer


# Load the model
# 1. Create config with shared embeddings enabled
config = TransformerConfig(shared_embeddings=True)

# 2. Load the checkpoint
checkpoint = torch.load('models/TNMT_v1_Beta_single.pt', weights_only=False, map_location='cpu')

# 3. Create model and move to device
model = Transformer(config)  # Use the modified config
device = "cuda:1"
model = model.to(device)
# 5. Load the modified state dict
model.load_state_dict(checkpoint['model_state_dict'])

tokenizer = Tokenizer(vocab_size=32000)
tokenizer_loaded = tokenizer.load_tokenizers('../Data/bi_tokenizers_32k')

translator = TranslationInference(
    model=model,
    tokenizer=tokenizer_loaded,
    device="cuda:1"
)


# Replace the interactive loop with evaluation mode when needed
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Translation Inference')
    parser.add_argument('--source', help='Source test file path')
    parser.add_argument('--target', help='Target test file path')
    parser.add_argument('--output', help='Output translations file path')
    
    args = parser.parse_args()

    bleu_score, _ = evaluate_bleu(
        translator=translator,
        source_file_path=args.source,
        target_file_path=args.target,
        output_file_path=args.output
    )
    print(f"BLEU Score: {bleu_score:.2f}")