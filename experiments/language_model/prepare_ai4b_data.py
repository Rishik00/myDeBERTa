# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm

def read_lines(file_path, start, end):
    """Lazy loads lines from a file within a given range."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= start:
                yield line.strip()

def get_data_split(split_type):
    """Defines start and end indices for dataset splits."""
    splits = {
        "train": (0, 1000),
        "validation": (1000, 2000),
        "test": (3000, 3500)
    }
    return splits.get(split_type, (0, 1000))  # Default to train if invalid

def tokenize_data(input_type, output_path, seq_length=512, vocab_id='deberta-v3-base', vocab_path=None):
    """Tokenizes and saves dataset based on the input type using lazy loading."""
    start, end = get_data_split(input_type)
    inp = read_lines('ai4b_mixed.txt', start, end)

    # Load tokenizer
    p, t = deberta.load_vocab(vocab_path=vocab_path, vocab_type='spm', pretrained_id=vocab_id)
    tokenizer = deberta.tokenizers[t](p)
    
    with open(output_path, 'w', encoding='utf-8') as wfs:
        for line in tqdm(inp, ncols=80, desc=f'Tokenizing {input_type} set'):
            tokens = tokenizer.tokenize(line) if line.strip() else []
            for idx in range(0, len(tokens), seq_length - 2):
                wfs.write(' '.join(tokens[idx:idx + seq_length - 2]) + '\n')

    print(f'Saved tokenized dataset to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset using DeBERTa tokenizer.")
    parser.add_argument('--input', required=True, help="Dataset split to use: 'train', 'validation', or 'test'.")
    parser.add_argument('--output', required=True, help="Output file path.")
    parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length.")
    parser.add_argument('--vocab_id', required=True, help='Model ID for vocabulary, e.g., mdeberta-v3-base')
    parser.add_argument('--vocab_path', help='Path to vocab file (optional)')
    
    args = parser.parse_args()
    
    # Default output filename if not provided
    if args.output is None:
        args.output = f"tokenized_{args.input}.txt"

    tokenize_data(args.input, args.output, args.max_seq_length, args.vocab_id, args.vocab_path)
