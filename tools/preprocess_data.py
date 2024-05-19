# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
import multiprocessing
import sys
import time

import pyarrow as pa
import torch
from datasets import load_dataset

from dolomite_engine.data.megatron.indexed_dataset import DType, MMapIndexedDatasetBuilder


class ArrowIterator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.fin = pa.ipc.open_file(args.input)
        self.num_records = self.fin.num_record_batches

    def __iter__(self):
        for i in range(self.num_records):
            doc = self.fin.get_batch(i)["tokens"].to_numpy().tolist()
            yield doc


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = "tokenizers/punkt/{}.pickle".format(self.args.lang)
            print("loading: " + library)
            splitter = nltk.load(library)
            Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def _encode_data(self, data):
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = self._encode_data(data)
        return ids, len(json_line)

    def encode_jsonl_zstd(self, bytes_obj):
        json_str = bytes_obj.decode("utf-8")
        return self.encode(json_str)

    def encode_hf(self, sample):
        ids = self._encode_data(sample)
        return ids, 1

    def encode_arrow(self, sample):
        if len(sample) > 0 and self.args.append_eod:
            sample.append(Encoder.tokenizer.eod)

        ids = {"text": [sample]}
        return ids, 0


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON/Arrow")
    group.add_argument(
        "--subset", type=str, default=None, help="Subset argument when loading input data from a HuggingFace dataset"
    )
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="space separate listed of keys to extract from json"
    )
    group.add_argument("--split-sentences", action="store_true", help="Split documents into sentences.")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "NullTokenizer",
            "TokenizerFromFile",
            "HuggingFaceTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument("--merge-file", type=str, default=None, help="Path to the BPE merge file (if necessary).")
    group.add_argument("--tokenizer-path", type=str, default=None, help="Path to the tokenizer")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")
    group.add_argument(
        "--lang", type=str, default="english", help="Language to use for NLTK-powered sentence splitting."
    )
    group.add_argument("--tokenizer-model", type=str, default=None, help="sentencepeice tokenizer model.")
    group.add_argument("--vocab-size", default=786, help="size of vocab for use with NullTokenizer")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")
    group.add_argument("--dataset-impl", type=str, default="mmap", choices=["lazy", "cached", "mmap"])

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, required=True, help="Number of worker processes to launch")
    group.add_argument("--chunk-size", type=int, required=True, help="Chunk size assigned to each worker process")
    group.add_argument("--log-interval", type=int, default=100, help="Interval between progress updates")
    args = parser.parse_args()

    if args.tokenizer_type.lower().startswith("bert"):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    print("Opening", args.input)

    if args.input.endswith(".jsonl"):
        print("Input is a jsonl file")
        assert args.subset is None, f"subset argument set to: {args.subset}, but loading a jsonl file."
        fin = open(args.input, "r", encoding="utf-8")
        encoded_docs = pool.imap(encoder.encode, fin, args.chunk_size)
        # encoded_docs = map(encoder.encode, fin)
    elif args.input.endswith(".jsonl.zst"):
        print("Input is a jsonl zst file")
        assert args.subset is None, f"subset argument set to: {args.subset}, but loading a zst jsonl file."
        import tempfile

        import zstandard

        dctx = zstandard.ZstdDecompressor()
        outfile = tempfile.TemporaryFile(suffix=args.input.rstrip(".zstd"))
        with open(args.input, "rb") as infile:
            dctx.copy_stream(infile, outfile)
        outfile.seek(0)
        encoded_docs = pool.imap(encoder.encode_jsonl_zstd, outfile, args.chunk_size)
    elif args.input.endswith(".arrow"):
        print("Input is an arrow file")
        assert args.subset is None, f"subset argument set to: {args.subset}, but loading an arrow file."
        fin = ArrowIterator(args)
        encoded_docs = pool.imap(encoder.encode_arrow, fin, args.chunk_size)
    else:
        # NOTE: this is not recommended for datasets larger than 40-50GB, as iterating through a dataset can be slow.
        # Somehow, it seems faster to first dump the dataset to a jsonl file: ds.to_json() and then process the jsonl file.
        # NOTE: this will be even slower if the dataset has large objects in other columns.
        # In this case, it is recommended to dump as json only the required key: ds = ds.remove_columns(...) then to_json()
        print("Input is not a jsonl file, will try to load from HF datasets")
        ds = load_dataset(args.input, use_auth_token=True, streaming=True, split="train", data_dir=args.subset)
        encoded_docs = pool.imap(encoder.encode_hf, ds, args.chunk_size)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key, level)
        builders[key] = MMapIndexedDatasetBuilder(
            output_bin_files[key], dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    total_docs = 0

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents", f"({i/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)

        total_docs += 1

    print("Done! Now finalizing.")
    open("{}_{}_{}.ndocs".format(args.output_prefix, key, level), "w").write(str(total_docs) + "\n")

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
