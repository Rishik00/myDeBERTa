import os
import glob
import re
import unicodedata
import random
import argparse
from tqdm import tqdm
import sentencepiece as spm
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description='Train a multilingual tokenizer on English and Hindi text')
    parser.add_argument('--en_data_dir', type=str, required=True, help='Directory containing English text files')
    parser.add_argument('--hi_data_dir', type=str, required=True, help='Directory containing Hindi text files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for tokenizer files')
    parser.add_argument('--vocab_size', type=int, default=128000, help='Vocabulary size for SentencePiece')
    parser.add_argument('--max_lines', type=int, default=10000000, help='Maximum number of lines to use for training')
    parser.add_argument('--balance_ratio', type=float, default=0.5, help='Ratio of Hindi to English text (0.5 means equal)')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of lines to sample from each language (if None, use max_lines)')
    return parser.parse_args()

def normalize_hindi_text(text):
    """Normalize Hindi text."""
    # Normalize Unicode characters
    text = unicodedata.normalize('NFC', text)
    
    # Handle specific Hindi normalization needs
    # Replace various forms of Hindi characters with canonical ones
    text = re.sub(r'\u0928\u094D', '\u0929', text)  # न् -> ऩ
    text = re.sub(r'\u0930\u094D', '\u0931', text)  # र् -> ऱ
    
    # Remove Zero Width Non-Joiner and Zero Width Joiner if not needed
    text = re.sub(r'[\u200C\u200D]', '', text)
    
    return text

def normalize_english_text(text):
    """Normalize English text."""
    # For English, we'll maintain case for mDeBERTa
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    return text

def clean_text(text, is_hindi=False):
    """Clean text by removing unwanted elements."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize punctuation
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'[–—−]', '-', text)
    
    # Handle specific language normalization
    if is_hindi:
        text = normalize_hindi_text(text)
    else:
        text = normalize_english_text(text)
    
    # Remove or replace invalid UTF-8 sequences
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    return text.strip()

def preprocess_file(file_path, is_hindi=False):
    """Preprocess a single file and return clean lines."""
    clean_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    cleaned = clean_text(line, is_hindi)
                    if cleaned:
                        clean_lines.append(cleaned)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    
    return clean_lines

def process_directory(directory, is_hindi=False, max_lines=None):
    """Process all text files in a directory."""
    all_lines = []
    files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    
    logger.info(f"Processing {'Hindi' if is_hindi else 'English'} files from {directory}")
    
    for file_path in tqdm(files, desc=f"{'Hindi' if is_hindi else 'English'} files"):
        lines = preprocess_file(file_path, is_hindi)
        all_lines.extend(lines)
    
    logger.info(f"Total {'Hindi' if is_hindi else 'English'} lines: {len(all_lines)}")
    
    if max_lines and len(all_lines) > max_lines:
        logger.info(f"Sampling {max_lines} lines from {len(all_lines)} {'Hindi' if is_hindi else 'English'} lines")
        return random.sample(all_lines, max_lines)
    
    return all_lines

def prepare_balanced_data(en_lines, hi_lines, balance_ratio=0.5, max_lines=None):
    """Prepare balanced dataset with specified Hindi to English ratio."""
    total_lines = len(en_lines) + len(hi_lines)
    
    if max_lines and total_lines > max_lines:
        hi_count = int(max_lines * balance_ratio)
        en_count = max_lines - hi_count
        
        # Ensure we don't request more lines than available
        hi_count = min(hi_count, len(hi_lines))
        en_count = min(en_count, len(en_lines))
        
        hi_sample = random.sample(hi_lines, hi_count)
        en_sample = random.sample(en_lines, en_count)
        
        combined = hi_sample + en_sample
    else:
        combined = hi_lines + en_lines
    
    random.shuffle(combined)
    logger.info(f"Final combined dataset: {len(combined)} lines")
    
    return combined

def train_sentencepiece(input_file, model_prefix, vocab_size=128000, input_sentence_size=10000000):
    """Train SentencePiece model with appropriate parameters for mDeBERTa."""
    logger.info(f"Training SentencePiece model with vocab size {vocab_size}")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # mDeBERTa uses BPE
        character_coverage=0.9995,  # High coverage for Hindi characters
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
        split_by_unicode_script=True,  # Important for mixed script (Devanagari + Latin)
        split_by_whitespace=True,
        bos_id=-1,  # No beginning of sentence token
        eos_id=-1,  # No end of sentence token
        pad_id=0,   # Set padding ID
        unk_id=1,   # Set unknown token ID
        user_defined_symbols=['[CLS]', '[SEP]', '[MASK]', '[PAD]'],  # Special tokens for mDeBERTa
    )
    logger.info("SentencePiece training completed")

def evaluate_tokenizer(model_path, test_sentences):
    """Evaluate the trained tokenizer on test sentences."""
    logger.info("Evaluating tokenizer")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    results = []
    
    for lang, sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        reconstructed = sp.decode(tokens)
        
        results.append({
            "language": lang,
            "original": sentence,
            "tokens": tokens,
            "token_count": len(tokens),
            "reconstructed": reconstructed,
            "is_perfect_reconstruction": sentence == reconstructed
        })
    
    return results

def main():
    args = setup_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process English and Hindi directories
    en_lines = process_directory(args.en_data_dir, is_hindi=False, max_lines=args.sample_size)
    hi_lines = process_directory(args.hi_data_dir, is_hindi=True, max_lines=args.sample_size)
    
    # Prepare balanced dataset
    combined_lines = prepare_balanced_data(
        en_lines, hi_lines, 
        balance_ratio=args.balance_ratio, 
        max_lines=args.max_lines
    )
    
    # Write combined lines to a temporary file for SentencePiece training
    input_file = os.path.join(args.output_dir, 'combined_input.txt')
    with open(input_file, 'w', encoding='utf-8') as f:
        for line in combined_lines:
            f.write(line + '\n')
    
    # Train SentencePiece model
    model_prefix = os.path.join(args.output_dir, 'mdeberta_tokenizer')
    train_sentencepiece(input_file, model_prefix, vocab_size=args.vocab_size)
    
    # Define test sentences for tokenizer evaluation
    test_sentences = [
        ('en', 'This is a test sentence in English.'),
        ('en', 'Hello, how are you doing today?'),
        ('hi', 'यह हिंदी में एक परीक्षण वाक्य है।'),
        ('hi', 'नमस्ते, आप आज कैसे हैं?'),
        ('mixed', 'This sentence has some हिंदी words mixed in between.')
    ]
    
    # Evaluate tokenizer
    evaluation_results = evaluate_tokenizer(f"{model_prefix}.model", test_sentences)
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, 'tokenizer_evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Tokenizer training and evaluation complete. Files saved to {args.output_dir}")
    
    # Remove temporary input file
    os.remove(input_file)

if __name__ == "__main__":
    main()