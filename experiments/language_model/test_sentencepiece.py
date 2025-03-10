## This script can test any sentencepiece model 
import time
import argparse
import sentencepiece as spm
import Levenshtein  # type:ignore

def count_unk_tokens(encoded_ids, tok):
    """Count how many unknown token IDs are in the encoded sequence"""
    return sum(1 for id in encoded_ids if id == tok.unk_id())

def levenshtein_error_rate(original, reconstructed):
    """Calculate character-level Levenshtein distance (edit distance) as an error rate"""
    if not original or not reconstructed:
        return 1.0  # Consider empty reconstructions as completely incorrect
    distance = Levenshtein.distance(original, reconstructed)
    return distance / max(len(original), len(reconstructed))

def test_sentencepiece(file_name, model_path, error_threshold=0.2):
    reconstruction_errors = 0
    unknown_tokens = 0
    total_tokens = 0
    high_error_sentences = []

    # Load model
    tok = spm.SentencePieceProcessor()
    tok.load(model_path)  # Correct way to load a model

    # Load test sentences
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    total_sentences = len(lines)

    for idx, line in enumerate(lines):
        # Encode and decode
        encoded = tok.encode(line)
        reconstructed = tok.decode(encoded)

        # Compute reconstruction error
        error_rate = levenshtein_error_rate(line, reconstructed)
        if error_rate > error_threshold:
            reconstruction_errors += 1
            high_error_sentences.append((idx, line, reconstructed, error_rate))

        # Count unknown tokens
        unk_count = count_unk_tokens(encoded, tok)
        unknown_tokens += unk_count
        total_tokens += len(encoded)

    # Calculate error rates
    reconstruction_error_rate = reconstruction_errors / total_sentences
    unk_token_rate = unknown_tokens / total_tokens if total_tokens > 0 else 0

    print(f'Sentences with high reconstruction errors more than {error_threshold * 100}: {reconstruction_errors} out of {total_sentences}')
    print(f'Reconstruction error rate: {reconstruction_error_rate:.4f}')
    print(f'Unknown tokens: {unknown_tokens} out of {total_tokens}')
    print(f'Unknown token rate: {unk_token_rate:.4f}')

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test any sentencepiece tokenizer with this script.")

    parser.add_argument('--input_file', required=True, help="Input file path")
    parser.add_argument('--tokenizer_model', default=None, help="Model path.")
    parser.add_argument('--error_threshold', type=int, default=512, help="Error threshold for levenshtein distance")

    args = parser.parse_args()

    test_sentencepiece(args.input_file, args.tokenizer_model, error_threshold=args.error_threshold)