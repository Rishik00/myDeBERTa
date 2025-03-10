# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm

def load_wikitext_data():
    """Loads the Wikitext-103 dataset from Hugging Face."""
    return load_dataset("wikitext", "wikitext-103-v1")

def load_fineweb_data():
    """Loads the Fineweb dataset from Hugging Face."""
    login('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')
    return load_dataset("KathirKs/fineweb-edu-hindi", "CC-MAIN-2014-52", trust_remote_code=True)

def tokenize_data(input_type, output_path=None, seq_length=512, vocab_id='deberta-v3-base', dataset_name='wikitext-103', vocab_path=None):
    """Tokenizes and saves dataset based on the input type."""
    
    # Load dataset
    if dataset_name == 'fineweb':
        data = load_fineweb_data()
        input_chunk = input_type if input_type in data else 'train'  # Preserve correct dataset split
        if input_chunk != input_type:
            print(f"[WARNING] {input_type} not found in fineweb dataset. Defaulting to 'train'.")
    elif dataset_name == 'ai4b':
        pass
    else:
        data = load_wikitext_data()
        if input_type not in data:
            raise ValueError(f"[ERROR] {input_type} is not a valid split for dataset '{dataset_name}'.")
        input_chunk = input_type  # No reassignment needed for wikitext

    print(f'Processing dataset split: {input_chunk}')
    inp = data[input_chunk]['text']

    # Handle slicing safely
    if input_type == 'train':
        print('here for train split')
        inp = inp[:1000]
    elif input_type == 'validation':
        print('here for valid strip')
        inp = inp[1000:2000]
    elif input_type == 'test':
        print('here for test strip')
        inp = inp[3000:3500]
    else:
        print(f"[WARNING] Not enough data for {input_chunk}, processing full dataset.")

    # Load tokenizer
    p, t = deberta.load_vocab(vocab_path=vocab_path, vocab_type='spm', pretrained_id=vocab_id)
    tokenizer = deberta.tokenizers[t](p)

    # Set output file name
    if output_path is None:
        output_path = f"{input_chunk}.spm"

    all_tokens = []
    for line in tqdm(inp, ncols=80, desc=f'Tokenizing {input_chunk} set'):
        tokens = tokenizer.tokenize(line) if line.strip() else []
        all_tokens.extend(tokens)

    print(f'Tokenized {len(all_tokens)} tokens from {input_chunk} dataset.')

    # Write tokenized data to file
    with open(output_path, 'w', encoding='utf-8') as wfs:
        for idx in range(0, len(all_tokens), seq_length - 2):
            wfs.write(' '.join(all_tokens[idx:idx + seq_length - 2]) + '\n')

    print(f'Saved tokenized dataset to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset using DeBERTa tokenizer.")

    parser.add_argument('-i', '--input', required=True, help="Dataset split to use: 'train', 'validation', or 'test'.")
    parser.add_argument('-o', '--output', default=None, help="Output file path.")
    parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length.")
    parser.add_argument('--dataset', required=True, help='Dataset name: wikitext-103 or fineweb')
    parser.add_argument('--vocab_id', required=True, help='Model ID for vocabulary, e.g., mdeberta-v3-base')
    parser.add_argument('--vocab_path', help='Path to vocab file (optional)')

    args = parser.parse_args()

    tokenize_data(args.input, args.output, args.max_seq_length, args.vocab_id, args.dataset, args.vocab_path)
