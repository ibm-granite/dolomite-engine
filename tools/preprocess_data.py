# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
import multiprocessing
import sys
import time
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dolomite_engine.data.megatron.indexed_dataset import DType, MMapIndexedDatasetBuilder


class Encoder(object):
    def __init__(self, tokenizer: AutoTokenizer, json_keys: List[str], append_eod: bool) -> None:
        self.tokenizer = tokenizer
        self.json_keys = json_keys
        self.append_eod = append_eod

    def _encode_data(self, data):
        ids = {}
        for key in self.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in [text]:
                sentence_ids = self.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.append_eod:
                doc_ids[-1].append(self.tokenizer.eos_token_id)
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

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer", type=str, default=None, help="Path to the tokenizer")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, required=True, help="Number of worker processes to launch")
    group.add_argument("--chunk-size", type=int, required=True, help="Chunk size assigned to each worker process")
    group.add_argument("--log-interval", type=int, default=100, help="Interval between progress updates")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    startup_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoder = Encoder(tokenizer, args.json_keys, args.append_eod)

    pool = multiprocessing.Pool(args.workers)
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
    else:
        # NOTE: this is not recommended for datasets larger than 40-50GB, as iterating through a dataset can be slow.
        # Somehow, it seems faster to first dump the dataset to a jsonl file: ds.to_json() and then process the jsonl file.
        # NOTE: this will be even slower if the dataset has large objects in other columns.
        # In this case, it is recommended to dump as json only the required key: ds = ds.remove_columns(...) then to_json()
        print("Input is not a jsonl file, will try to load from HF datasets")
        ds = load_dataset(args.input, use_auth_token=True, streaming=True, split="train", data_dir=args.subset)
        encoded_docs = pool.imap(encoder.encode_hf, ds, args.chunk_size)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = f"{args.output_prefix}_{key}.bin"
        output_idx_files[key] = f"{args.output_prefix}_{key}.idx"
        builders[key] = MMapIndexedDatasetBuilder(
            output_bin_files[key], dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

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

    print("Done! Now finalizing.")

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
