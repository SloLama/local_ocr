from merge_pages import process_document as merge_document
from classify_pages import process_document as classify_document

import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser
import math


def postprocess_corpus(input_dir, output_dir, log_dir, model_dict, skip_existing, n_chunks, chunk_index):
    documents = [subdir for subdir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, subdir))]
    n_documents = len(documents)

    if n_chunks > 1:
        documents.sort()
        chunk_size = int(math.ceil(n_documents / n_chunks))
        start_index = chunk_index * chunk_size
        print("Chunk start index:", start_index)
        end_index = min(start_index + chunk_size, n_documents)
        print("Chunk end index:", end_index)

        documents = documents[start_index:end_index]
        n_documents = end_index - start_index

    for i, document in enumerate(documents):
        input_path = os.path.join(input_dir, document)
        output_path = os.path.join(output_dir, document + ".md")
        if skip_existing and os.path.exists(output_path):
            continue
        classify_log_path = os.path.join(log_dir, document + "_classify.jsonl")
        merge_log_path = os.path.join(log_dir, document + "_merge.jsonl")

        print(f"Processing document {i+1} of {n_documents}")

        page_list = classify_document(input_path, classify_log_path, **model_dict)
        merged_doc = merge_document(page_list, merge_log_path, **model_dict)

        if merged_doc is None:
            continue

        f_out = open(output_path, "w")
        f_out.write(merged_doc)
        f_out.close()

    print("Done!")


def create_model_dict(args):
    model = LLM(
        args.model_path,
        seed=42,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=32768,
        dtype="bfloat16"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=8192
    )

    model_dict = {
        "model": model,
        "tokenizer": tokenizer,
        "sampling_params": sampling_params,
        "batch_size": args.batch_size
    }

    return model_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (either HF URI or local path)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input dir, containing subdirs that represent separate documents"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output_dir, where processed markdowns will be stored"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the log_dir, where postprocessing model predictions are stored."
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Number of GPUs to split model across using tensor parallelism."
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="Percentage of GPU memory to use for model weights and KV cache."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Inference batch size."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If specified, files that already exist are not processed again."
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=1,
        help="Number of chunks to separate input data into."
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=0,
        help="Index of the chunk to be processed. Should be integer between 0 and n_chunks-1."
    )
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    model_dict = create_model_dict(args)
    postprocess_corpus(args.input_dir, args.output_dir, args.log_dir, model_dict, args.skip_existing, args.n_chunks, args.chunk_index)
