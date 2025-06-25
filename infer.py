import torch
from tokenizer import BilingualTokenizer as Tokenizer
from model import Transformer, TransformerConfig

from translator import TranslationInference



# 1. Create config with shared embeddings enabled
config = TransformerConfig(shared_embeddings=True)

# 2. Load the checkpoint
checkpoint = torch.load('models/TNMT_v1_Beta_single.pt', weights_only=False, map_location='cpu')

# 3. Create model and move to device
model = Transformer(config)  # Use the modified config
device = "cpu"
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

# Translation inside loop
while True:
    source_text = input("Enter text to translate (or 'exit' to quit): ")
    if source_text.lower() == 'exit':
        break
    translated_text = translator.translate_text(source_text)
    print(f"Translated text: {translated_text}")