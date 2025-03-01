import os
import re
import sentencepiece as sp


class MinSPMTokenizer:
    def __init__(self, vocab_path, do_lowr=False, special_tokens=None, bpe_dropout=0, split_by_punct=False):
        self.split_by_punct = split_by_punct

        assert os.path.exists(vocab_path)

        spm = sp.SentencePieceProcessor()
        spm.load(vocab_path)
        bpe_vocab_size = spm.GetPieceSize()

        self.vocab = {spm.IdToPiece(i):i for i in range(bpe_vocab_size)}
        self.id_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        self.spm = spm
        self.pattern =  re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        _special_tokens = ['[MASK]', '[SEP]', '[PAD]', '[UNK]', '[CLS]']
        self._special_tokens = [] if special_tokens is None else special_tokens

        for t in _special_tokens: self.add_special_tokens(t)


    def tokenize(self):
        pass

    def add_special_tokens(self):
        pass

    def tokens_to_ids(self, tokens):
        pass

    def ids_to_tokens(self, ids):
        pass

    def symb(self, id): 
        pass
    
    def encode_pieces(self, text):
        pass

    def split_to_words(self, text):
        pass

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        pass

    def add_special_tokens(self, token):
        pass


    def pad(self): return '[PAD]'
    def box(self): return '[BOS]'
    def eos(self): return '[EOS]'
    def unk(self): return '[UNK]'
    def mask(self): return '[MASK]'

