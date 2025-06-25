import torch
from tqdm import tqdm
import sacrebleu
from typing import List, Tuple, Optional

def evaluate_bleu(translator, source_file_path: str, target_file_path: str, 
                  output_file_path: Optional[str] = None) -> Tuple[float, List[str]]:
    """
    Evaluate translation quality using SacreBLEU score
    
    Args:
        translator: TranslationInference instance
        source_file_path: Path to file with source language sentences
        target_file_path: Path to file with target language sentences (references)
        output_file_path: Optional path to save translations
        
    Returns:
        BLEU score and list of translations
    """
    # Load source and target sentences
    with open(source_file_path, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f.readlines()]
    
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_sentences = [line.strip() for line in f.readlines()]
    
    # Make sure we have matching number of source and target sentences
    assert len(source_sentences) == len(target_sentences), "Source and target files must have same number of sentences"
    
    print(f"Loaded {len(source_sentences)} sentences for evaluation")
    
    # Translate source sentences
    translations = []
    for sentence in tqdm(source_sentences, desc="Translating"):
        translation = translator.translate_text(sentence)
        translations.append(translation)
    
    # Save translations if output path is provided
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for trans in translations:
                f.write(trans + '\n')
        print(f"Saved translations to {output_file_path}")
    
    # Calculate BLEU score using sacrebleu
    # SacreBLEU expects a list of references for each hypothesis
    refs = [target_sentences]  # List of references (just one reference per translation in this case)
    bleu = sacrebleu.corpus_bleu(translations, refs)
    
    return bleu.score, translations


