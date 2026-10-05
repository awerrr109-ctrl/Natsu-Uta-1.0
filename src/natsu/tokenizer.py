"""
Tokenizers.
* ByteTokenizer (data.py) — toy stages.
* BPE (HF `tokenizers`, byte-level, trained streaming from a text file) — removes the byte-level confound from
  memory-module results (Engram learns n-gram statistics; at byte level local statistics are a large part of the loss).
  Final 9B spec: 131,072 vocab trained on the pretraining mixture (TRAINING_SPEC §3). Toy: 4,096.
Specials: <pad>=0, <bos>=1, <eos>=2.
"""
import os
import numpy as np


def train_bpe(src_txt, out_json, vocab=4096, max_bytes=60_000_000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<pad>", "<bos>", "<eos>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def it():
        n = 0
        with open(src_txt, encoding="utf-8", errors="replace") as f:
            for doc in f.read(max_bytes).split("<|endoftext|>"):
                n += len(doc)
                yield doc
    tok.train_from_iterator(it(), tr)
    tok.save(out_json)
    return tok


class BPETokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)
        self.vocab_size = self.t.get_vocab_size()
        self.bos, self.eos, self.pad = 1, 2, 0

    def encode(self, s, bos=False, eos=False):
        ids = self.t.encode(s).ids
        return ([self.bos] if bos else []) + ids + ([self.eos] if eos else [])

    def decode(self, ids):
        return self.t.decode([i for i in ids if i > 2])


def encode_file_bpe(src, dst, tok, chunk_bytes=1 << 22, doc_sep="<|endoftext|>"):
    """Streaming encode to uint16 memmap. Returns (n_tokens, n_bytes) so bits-per-BYTE stays comparable across tokenizers."""
    n_tok = n_bytes = 0
    with open(src, encoding="utf-8", errors="replace") as f, open(dst, "wb") as out:
        buf = ""
        while True:
            s = f.read(chunk_bytes)
            if not s:
                break
            buf += s
            docs = buf.split(doc_sep)
            buf = docs.pop()
            docs = [d.strip() for d in docs if d.strip()]
            for enc, d in zip(tok.t.encode_batch(docs), docs):
                a = np.asarray([tok.bos] + enc.ids, dtype=np.uint16)
                a.tofile(out)
                n_tok += a.size
                n_bytes += len(d.encode("utf-8")) + 1
    return n_tok, n_bytes
