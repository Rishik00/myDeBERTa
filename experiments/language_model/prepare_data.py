# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm

def load_wikitext_data():
    """Loads the Wikitext-103 dataset from Hugging Face."""
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    return dataset

def load_fineweb_data():
    login('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')
    dataset = load_dataset("KathirKs/fineweb-edu-hindi", "CC-MAIN-2014-52", trust_remote_code=True)
    return dataset

def tokenize_data(input_type, output_path=None, seq_length=512):
    """Tokenizes and saves Wikitext data."""
    data = load_fineweb_data()  

    # Check for valid input type
    # if input_type not in data:
    #     raise ValueError(f"Invalid dataset split: {input_type}. Choose from 'train', 'validation', or 'test'.")

    inp = data['train']['text']
    if input_type == 'train':
        inp = inp[:1000]
    elif input_type == 'validation':
        inp = inp[1000:2000]
    else:
        inp = inp[3000:3500]

    # Load tokenizer
    p, t = deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id='mdeberta-v3-base')
    tokenizer = deberta.tokenizers[t](p)

    # Set output file name
    if output_path is None:
        output_path = f"{input_type}.spm"

    all_tokens = []
    for line in tqdm(inp, ncols=80, desc=f'Loading {input_type} set'):
        tokens = tokenizer.tokenize(line) if line.strip() else []
        all_tokens.extend(tokens)

    print(f'Loaded {len(all_tokens)} tokens from {input_type} dataset.')

    # Write tokenized data
    lines = 0
    with open(output_path, 'w', encoding='utf-8') as wfs:
        idx = 0
        while idx < len(all_tokens):
            wfs.write(' '.join(all_tokens[idx:idx+seq_length-2]) + '\n')
            idx += (seq_length - 2)
            lines += 1

    print(f'Saved {lines} lines to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize Wikitext-103 using DeBERTa tokenizer.")

    parser.add_argument('-i', '--input', required=True, help="Dataset split to use: 'train', 'validation', or 'test'.")
    parser.add_argument('-o', '--output', default=None, help="Output file path.")
    parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length.")
    # parser.add_argument('-dp', '--dataset_path', default=None, help="input dataset path.")
    args = parser.parse_args()
    tokenize_data(args.input, args.output, args.max_seq_length)
