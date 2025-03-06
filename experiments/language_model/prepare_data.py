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
    """Loads the Fineweb dataset from Hugging Face."""

    login('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')
    dataset = load_dataset("KathirKs/fineweb-edu-hindi", "CC-MAIN-2014-52", trust_remote_code=True)
    return dataset

def tokenize_data(input_type, output_path=None, seq_length=512, vocab_id='deberta-v3-base', dataset_name='wikitext-103', vocab_path=None):
    """Tokenizes and saves dataset based on the input type."""
    
    # Load the appropriate dataset based on input
    input_chunk = None
    if dataset_name == 'fineweb':
        data = load_fineweb_data()
        # Default to 'train' split if input_type is not 'train'
        if input_type not in ['train']:
            input_chunk = 'train'
    
    else:
        data = load_wikitext_data()
        input_chunk = input_type

    print(f'input type: {input_type}')
    inp = data[input_chunk]['text']

    # Handle slicing of input if necessary
    if input_type == 'train':
        print("Entering here for training")
        inp = inp[:1000]
    elif input_type == 'validation':
        print("Entering here for validation")
        inp = inp[1000:2000]
    else:
        print("Entering for test set")
        inp = inp[3000:3500]

    # Load tokenizer
    p, t = deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id=vocab_id)
    tokenizer = deberta.tokenizers[t](p)

    # Set output file name
    if output_path is None:
        output_path = f"{input_type}.spm"

    all_tokens = []
    for line in tqdm(inp, ncols=80, desc=f'Loading {input_type} set'):
        tokens = tokenizer.tokenize(line) if line.strip() else []
        all_tokens.extend(tokens)

    print(f'Loaded {len(all_tokens)} tokens from {input_type} dataset.')

    # Write tokenized data to file
    lines = 0
    with open(output_path, 'w', encoding='utf-8') as wfs:
        idx = 0
        while idx < len(all_tokens):
            wfs.write(' '.join(all_tokens[idx:idx + seq_length - 2]) + '\n')
            idx += (seq_length - 2)
            lines += 1

    print(f'Saved {lines} lines to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset using DeBERTa tokenizer.")

    # Add arguments for the script
    parser.add_argument('-i', '--input', required=True, help="Dataset split to use: 'train', 'validation', or 'test'.")
    parser.add_argument('-o', '--output', default=None, help="Output file path.")
    parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length.")
    parser.add_argument('--dataset', help='Name of the dataset for preprocessing')
    parser.add_argument('--vocab_id', help='Name of the model for vocab, ex: mdeberta-v3-base')
    parser.add_argument('--vocab_path',help='Path of the model vocab for vocab, ex: mdeberta-v3-base')
    # Parse the arguments
    args = parser.parse_args()

    # Call the function to tokenize data
    tokenize_data(args.input, args.output, args.max_seq_length, args.vocab_id, args.dataset, args.vocab_path)
