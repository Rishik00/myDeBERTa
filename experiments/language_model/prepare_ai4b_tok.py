import re, time
import unicodedata
import argparse
from tqdm import tqdm
import sentencepiece as spm
from typing import List
from datasets import load_dataset

class HindiEnglishCleaner:
    def __init__(self, lang):
        self.lang = lang
        
    def split_row(self, row):
        row = row.split('.') if self.lang == 'en' else row.split('।')
        return [sentence.strip() for sentence in row if sentence.strip()]

    def normalize(self, text):
        if self.lang == 'en': text = unicodedata.normalize('NFKC', text)
        else: text = unicodedata.normalize('NFC', text)
        return text
    
    def clean(self, text):
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Normalize punctuation
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[–—−]', '-', text)
        return text

def clean_text(row: str, lang: str):
    cleaner = HindiEnglishCleaner(lang=lang)
    cleaned_text = cleaner.clean(row)
    cleaned_text = cleaner.normalize(cleaned_text)

    return cleaner.split_row(cleaned_text)

def save_file(inp_file, dataset, lang, num_rows):
    with open(inp_file, 'w', encoding='urf-8') as f:
        count = 0
        for idx, sample in enumerate(dataset['train']):

            if idx >= num_rows:
              break
            is_hindi = lang == 'hi'
            cleaned_text = clean_text(sample['text'], is_hindi)

            if lang == 'en' and len(cleaned_text.split(' ')) >= 3:
                for sentence in cleaned_text:
                    f.write(sentence + "\n")
                    count += 1

            if lang == 'hi' and len(cleaned_text.split(' ')) >= 1:
                for sentence in cleaned_text:
                    f.write(sentence + "\n")
                count += 1

    print(f'Written {count} rows into {inp_file}')
    return count

def mix_files(en_file, hi_file, ofile):
    final_file_name = ofile if ofile is not None else 'default.txt'

    with open(en_file, 'r', encoding='utf-8') as enf, open(hi_file, 'r', encoding='utf-8') as hif:
        eng_sentences = enf.readlines()
        hin_sentences = hif.readlines()

    # Get the minimum length to prevent index errors
    min_length = min(len(eng_sentences), len(hin_sentences))

    with open(final_file_name, 'w', encoding='utf-8') as ofile:
        for i in range(min_length):
            ofile.write(eng_sentences[i].strip() + '\n' + hin_sentences[i].strip() + '\n')

    print(f'Done writing sentences to {final_file_name}')
    print(f"Total English sentences: {len(eng_sentences)}")
    print(f"Total Hindi sentences: {len(hin_sentences)}")


def train_sentencepiece(input_file, model_prefix, vocab_size=10000):
    """Train SentencePiece model with appropriate parameters for mDeBERTa."""
    print(f"Training SentencePiece model with vocab size {vocab_size}")
    start_time = time.time()
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
        split_by_unicode_script=True,
        control_symbols=['[CLS]', '[SEP]', '[MASK]', '[PAD]']  # Correct way to add special tokens
    )
    end_time = time.time()
    print(f"SentencePiece training completed in {end_time-start_time} s")


if __name__ == "__main__":
    train_files = ['ai4b_en.txt', 'ai4b_hi.txt', 'ai4b_mixed.txt']
    parser = argparse.ArgumentParser(description="Tokenize dataset using a custom trained tokenizer.")

    parser.add_argument('--num_en', default=None, type=int, help="Limit number of English sentences")
    parser.add_argument('--num_hi', default=None, type=int, help="Limit number of Hindi sentences")
    parser.add_argument('--streaming', default=True, type=bool, help="Enable streaming for datasets")
    parser.add_argument('--verified', default=True, type=bool, help="Use verified dataset version")
    parser.add_argument('--vocab_size', default=10000, type=int, help="Vocab size for SentencePiece")
    parser.add_argument('--model_type', default='bpe', choices=['bpe', 'unigram'], help="SentencePiece model type")
    args = parser.parse_args()

    print(f'Loading dataset: sangraha english')
    en_dataset = load_dataset("ai4bharat/sangraha", data_dir="verified/eng", streaming=args.streaming)

    print(f'Loading dataset: sangraha hindi')
    hi_dataset = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", streaming=args.streaming)

    ## Change num_rows for getting more data for train set.
    num_en = save_file(train_files[0], en_dataset, 'en', num_rows=args.num_en)
    num_hi = save_file(train_files[1], hi_dataset, 'hi', num_rows=args.num_hi)

    mix_files(train_files[0], train_files[1], train_files[2])

    train_sentencepiece(input_file=train_files[2], model_prefix='mixed', vocab_size=args.vocab_size)
