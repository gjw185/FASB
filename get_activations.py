import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from FASB_utils import get_llama_activations_bau, tokenized_tqaa, tokenized_tqa_gen_end_q_pd
import llama
import pickle
import argparse
import pdb
import json
import pandas as pd

def main():
    """
    Specify dataset name as the first command line argument. Current options are
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the
    validation set for the specified dataset on the last token for llama-7B.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()

    if args.model_name == "llama2_chat_7B" or args.model_name == 'std_llama2_chat_7B':
        MODEL = '/root/gjw/transformer/Llama-2-7b-chat-hf'
    elif args.model_name == "llama_7B" or args.model_name == 'std_llama_7B':
        MODEL = '/root/gjw/transformer/Llama-2-7b-hf'
    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,device_map="auto")

    device = "cuda"

    # pdb.set_trace()
    if args.dataset_name == "tqa_mc2":
        with open("/root/gjw/honest_llama/TruthfulQA/data/mc_task.json", "r") as f:
            dataset = json.load(f)
        formatter = tokenized_tqaa
    elif args.dataset_name == 'tqa_gen_end_q':
        dataset = pd.read_csv('./TruthfulQA/TruthfulQA.csv')
        formatter = tokenized_tqa_gen_end_q_pd

    # pdb.set_trace()
    print("Tokenizing prompts")
    prompts, labels, _ = formatter(dataset, tokenizer)

    all_head_wise_activations = []
    print("Getting activations")
    # pdb.set_trace()
    for prompt in tqdm(prompts):
        _, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        all_head_wise_activations.append(head_wise_activations[:, -1, :])

    print("Saving labels")
    np.save(f'features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving head wise activations")
    np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)



if __name__ == '__main__':
    main()