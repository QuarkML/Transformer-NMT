# Clean the text
# Convert english to lower case
# Remove punctuation
# Remove unicode characters that are not being rendered properly

import re
from typing import List, Tuple
import unicodedata

from typing import List, Tuple
import unicodedata
import re
from collections import Counter

def preprocess_text(en_sentences: List[str], hi_sentences: List[str]) -> Tuple[List[str], List[str]]:
    """
    Preprocess parallel English-Hindi text by cleaning Unicode characters, normalizing text,
    and removing duplicate sentence pairs.
    
    Args:
        en_sentences: List of English sentences
        hi_sentences: List of Hindi sentences
        
    Returns:
        Tuple[List[str], List[str]]: Cleaned English and Hindi sentences with duplicates removed
    """
    def clean_unicode(text: str) -> str:
        """Remove unwanted Unicode characters and normalize Unicode forms."""
        # Normalize Unicode to compose characters where possible
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width joiner and non-joiner
        text = re.sub(r'[\u200c\u200d]', '', text)
        
        # Remove other invisible Unicode characters
        text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def clean_english(text: str) -> str:
        """Clean and normalize English text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        return text.strip()
    
    def clean_hindi(text: str) -> str:
        """Clean Hindi text while preserving valid Hindi Unicode characters."""
        # Clean Unicode issues
        text = clean_unicode(text)
        
        # Additional Hindi-specific cleaning if needed
        # Remove any ASCII punctuation that should be Hindi punctuation
        text = text.replace('?', '?').replace(',', '،')
        
        return text.strip()
    
    # Process all sentences and track duplicates
    processed_en = []
    processed_hi = []
    dropped_count = 0
    duplicate_count = 0
    
    # Create a list of tuples for sentence pairs
    sentence_pairs = []
    
    for en, hi in zip(en_sentences, hi_sentences):
        # Clean the sentences
        clean_en = clean_english(en)
        clean_hi = clean_hindi(hi)
        
        # Only keep pairs where both sentences are non-empty after cleaning
        if clean_en and clean_hi:
            sentence_pairs.append((clean_en, clean_hi))
        else:
            dropped_count += 1
    
    # Count occurrences of each unique sentence pair
    pair_counts = Counter(sentence_pairs)
    
    # Keep track of duplicate statistics
    duplicate_pairs = {pair: count for pair, count in pair_counts.items() if count > 1}
    
    # Keep only unique pairs (first occurrence)
    unique_pairs = list(dict.fromkeys(sentence_pairs))
    
    # Separate the pairs back into English and Hindi lists
    processed_en, processed_hi = zip(*unique_pairs) if unique_pairs else ([], [])
    processed_en, processed_hi = list(processed_en), list(processed_hi)
    
    # Calculate statistics
    total_duplicates_removed = sum(count - 1 for count in pair_counts.values() if count > 1)

    # Print the statistics when needed,
    # it might be annoying when we do it during testing

    # print(f"Preprocessing Stats:")
    # print(f"Original sentence pairs: {len(en_sentences)}")
    # print(f"Processed sentence pairs: {len(processed_en)}")
    # print(f"Dropped empty/invalid pairs: {dropped_count}")
    # print(f"Duplicate pairs removed: {total_duplicates_removed}")
    # print(f"Number of unique pairs that had duplicates: {len(duplicate_pairs)}")
    
    # Print some examples of duplicates that were removed (up to 3 examples)
    # if duplicate_pairs:
    #     print("\nExample duplicate pairs that were consolidated:")
    #     for (en, hi), count in list(duplicate_pairs.items())[:3]:
    #         print(f"'{en}' ⟷ '{hi}' (appeared {count} times)")
    
    return processed_en, processed_hi


def split_sequences(tokenized_en, tokenized_hi, max_seq_length = 256):
    
    """ The function is used for splitting the larger sequences that are greater than the max_seq_length """
    
    processed_en = []
    processed_hi = []


    for en_seq, hi_seq in zip(tokenized_en, tokenized_hi):
        en_len = len(en_seq)
        hi_len = len(hi_seq)


        # If both sequences are within the max length, keep them as is
        if en_len <= max_seq_length and hi_len <= max_seq_length:
            processed_en.append(en_seq)
            processed_hi.append(hi_seq)
            continue
        
        # Finding the number of splits that is needed
        max_len = max(en_len, hi_len) # Find the maximum length of the two sequences

        # Calculate the number of splits needed,
        # This willl give the number of splits needed to split the sequence into max_seq_length
        # Not the most efficient way to split since it splits to the smallest possible size
        n_splits = (max_len + max_seq_length - 1) // max_seq_length

        # Finding better split size
        if max_len % n_splits == 0: # If the max_len is divisible by n_splits, then we can split the sequence into equal parts
            split_size = max_len // n_splits
        else:

            # Otherwise, we need to find the largest divisor that creates segments <= max_seq_length
            for i in range(n_splits, max_len):
                if max_len % i == 0 and max_len // i <= max_seq_length:
                    n_splits = i
                    split_size = max_len // n_splits
                    break
            else:
                split_size = (max_len + n_splits - 1) // n_splits


        # Split the sequences based on the calculated split size
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx_en = min((i + 1) * split_size, en_len)
            end_idx_hi = min((i + 1) * split_size, hi_len)

            en_subseq = en_seq[start_idx:end_idx_en]
            hi_subseq = hi_seq[start_idx:end_idx_hi]


            # Since there is chance that the splitted sequence is still greater than max_seq_length
            # We need to split it again, this is a recursive call that does it until the sequence is less than max_seq_length
            if len(en_subseq) > max_seq_length or len(hi_subseq) > max_seq_length:
                sub_en, sub_hi = split_sequences([en_subseq], [hi_subseq], max_seq_length)
                processed_en.extend(sub_en)
                processed_hi.extend(sub_hi)
            else:
                processed_en.append(en_subseq)
                processed_hi.append(hi_subseq)

    return processed_en, processed_hi
