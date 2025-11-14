import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import pdb
import sys
import json

sys.path.append('../../')
from FASB_utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, \
    get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default='llama_7B', help='model name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model',
                        default=False)
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q',
                        help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--back_num', type=int, default=10, help='back_num')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--used_ratio', type=float, default=1.0, help='used_ratio')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size',
                        default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('../TruthfulQA/TruthfulQA.csv')

    # order csv by huggingface order, the order used to save activations
    with open("/root/gjw/honest_llama/TruthfulQA/data/mc_task.json", "r") as f:
        dataset = json.load(f)
    # dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
    golden_q_order = []
    for i in range(len(dataset)):
        question = dataset[i]["question"]
        golden_q_order.append(question)
    # golden_q_order = list(dataset["question"])
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)}))

    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)

    # pdb.set_trace()
    if args.model_name == 'llama_7B':
        MODEL = '/root/gjw/transformer/Llama-2-7b-hf'
    elif args.model_name == 'llama2_chat_7B':
        MODEL = '/root/gjw/transformer/Llama-2-7b-chat-hf'

    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                   device_map="auto")

    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations
    head_wise_activations = np.load(f"../features/{args.model_name}_{args.dataset_name}_head_wise.npy")
    labels = np.load(f"../features/{args.model_name}_{args.dataset_name}_labels.npy")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    tuning_activations = np.load(f"../features/std_{args.model_name}_{activations_dataset}_head_wise.npy")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h=num_heads)

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels,
                                                                                                    head_wise_activations)

    # run k-fold cross validation
    results = []
    # pdb.set_trace()
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * (1 - args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"splits/_fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"splits/_fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_idxs].to_csv(f"splits/_fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get heads
        top_heads, probes, top_heads_id = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations,
                                                        separated_labels, num_layers, num_heads, args.seed,
                                                        args.num_heads, args.use_random_dir)

        print("Heads intervened: ", sorted(top_heads))

        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs,
                                                separated_head_wise_activations, separated_labels)
        else:
            com_directions = None

        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads,
                                               args.use_center_of_mass, args.use_random_dir, com_directions)

        def lt_modulated_vector_add(head_output, layer_name, pro=1.0, start_edit_location='lt'):

            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                if start_edit_location == 'lt':
                    head_output[:, -1, head, :] += args.alpha * pro * proj_val_std * direction_to_add
                else:
                    head_output[:, start_edit_location:, head, :] += args.alpha * pro * proj_val_std * direction_to_add
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output

        filename = f'ratio_{args.used_ratio}_after_back_{args.back_num}_thr_{args.threshold}_{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'

        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'
        if args.use_honest:
            filename = 'honest_' + filename

        curr_fold_results = alt_tqa_evaluate(
            args.used_ratio,
            args.alpha,
            args.back_num,
            args.threshold,
            top_heads,
            top_heads_id,
            probes,
            {args.model_name: model},
            ['judge', 'info', 'mc'],
            f"splits/_fold_{i}_test_seed_{args.seed}.csv",
            f'results_dump/probe_later_back_answer_dump/{filename}.csv',
            f'results_dump/probe_later_back_summary_dump/{filename}.csv',
            device="cuda",
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add,
            judge_name=args.judge_name,
            info_name=args.info_name
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)

    results = np.array(results)
    final = results.mean(axis=0)

    print(
        f'True*Info Score: {final[1] * final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}')
    #

if __name__ == "__main__":
    main()
