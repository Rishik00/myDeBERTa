from DeBERTa import deberta
import argparse
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm

def load_data_via_streaming():
    num_en_train = 1210
    num_hi_train = 1650

    file_name = 'mixeed.txt'
    temp_files = ['en.txt', 'hi.txt']

    en_data = load_dataset("allenai/c4", 'en', split='train', streaming=True)
    with open(temp_files[0], 'w') as enfile:
        for i, sample in enumerate(en_data):
            if i >= num_en_train:
                break
            cleaned_row = sample['text'].split('.')
            for row in cleaned_row:
                if len(row.split()) >= 3:
                    enfile.write(row + '\n')
 
    hi_data = load_dataset("zicsx/C4-Hindi-Cleaned", split='train', streaming=True)
    with open(temp_files[1], 'w') as hifile:
        for i, sample in enumerate(hi_data):
            if i >= num_hi_train:
                break
            cleaned_row = sample['text'].split('।')
            for row in cleaned_row:
                if len(row.split()) >= 3:
                    hifile.write(row + '\n')

    with open(temp_files[0], 'r') as file0, open(temp_files[1], 'r') as file1:
        eng_sentences = file0.readlines()
        hin_sentences = file1.readlines()

    with open(file_name, 'w') as ofile:
        for eng, hin in zip(eng_sentences, hin_sentences):
            ofile.write(eng.strip() + '\n')
            ofile.write(hin.strip() + '\n')

    return file_name


def tokenize_data(split_name, output_path=None, seq_length=512, vocab_id='deberta-v3-base'):
    file_name = load_data_via_streaming()
    with open(file_name, 'r') as f:
        inp = f.readlines()

    if split_name == 'train':
        inp = inp[:1000]
    elif split_name == 'validation':
        inp = inp[1000:2000]
    else:
        inp = inp[3500:5000]
    
    p, t = deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id=vocab_id)
    tokenizer = deberta.tokenizers[t](p)

    # Set output file name
    if output_path is None:
        output_path = f"{split_name}.spm"

    all_tokens = []
    for line in tqdm(inp, ncols=80, desc=f'Loading {split_name} set'):
        tokens = tokenizer.tokenize(line) if line.strip() else []
        all_tokens.extend(tokens)

    print(f'Loaded {len(all_tokens)} tokens from {split_name} dataset.')

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

    args = parser.parse_args()

    # Call the function to tokenize data
    tokenize_data(args.input, args.output, args.max_seq_length, args.vocab_id)
