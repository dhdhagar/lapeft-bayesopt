import os
import time
import json
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm
from sklearn.utils import shuffle as skshuffle
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

# Non-finetuned surrogates. The finetuned surrogates are in lapeft_bayesopt.surrogates
from fixed_feat_surrogate import LaplaceBoTorch
from lapeft_bayesopt.foundation_models.t5 import T5Regressor
from lapeft_bayesopt.foundation_models.llama2 import Llama2Regressor
from lapeft_bayesopt.foundation_models.utils import get_llama2_tokenizer, get_t5_tokenizer
from lapeft_bayesopt.utils.acqf import thompson_sampling
from lapeft_bayesopt.utils import helpers
# Our self-defined problems, using the format provided by lapeft-bayesopt
from data_processor import TwentyQuestionsDataProcessor
from prompting import MyPromptBuilder


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--data_dir", type=str, default='data/twentyquestions/datasets/word2vec'
        )
        self.add_argument(
            "--cache_dir", type=str, default='cache'
        )
        self.add_argument(
            "--out_dir", type=str, default="outputs"
        )
        self.add_argument(
            "--dataset", type=str
        )
        self.add_argument(
            "--test_idx_or_word", type=str
        )
        self.add_argument(
            "--prompt_strategy", type=str, choices=['word', 'instruction', 'hint', 'hint-goodness'], default='word'
        )
        self.add_argument(
            "--hint", type=str, default=""
        )
        self.add_argument(
            "--model", type=str,
            choices=['t5-small', 't5-base', 't5-large', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b'],
            default="llama-2-7b"
        )
        self.add_argument(
            "--init_strategy", type=str, choices=['random'], default="random"
        )
        self.add_argument(
            "--acquisition_fn", type=str, choices=['thompson_sampling'], default="thompson_sampling"
        )
        self.add_argument(
            "--cuda", action=argparse.BooleanOptionalAction, default=True
        )
        self.add_argument(
            "--rescale_scores", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--save_word_specific_dataset", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--feat_extraction_strategy", type=str, choices=['last-token', 'average', 'max', 'first-token'],
            default="last-token"
        )
        self.add_argument(
            "--exit_after_feat_extraction", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--reset_cache", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--seed", type=int, default=9999
        )
        self.add_argument(
            "--n_init_data", type=int, default=5
        )
        self.add_argument(
            "--T", type=int, default=100
        )
        self.add_argument(
            "--n_seeds", type=int, default=1
        )
        self.add_argument(
            "--debug", action=argparse.BooleanOptionalAction, default=False
        )


def get_avg_features(feat, data):
    # feat: (batch_size, seq_len, hidden_size)
    # this is the last_hidden_state;
    # this includes padding tokens, so we need to mask them
    mask = data.attention_mask.to(feat.device)
    feat = (feat * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return feat


def get_max_features(feat, data):
    # feat: (batch_size, seq_len, hidden_size)
    # this is the last_hidden_state;
    # this includes padding tokens, so we need to mask them
    mask = data.attention_mask.to(feat.device) * 10 - 9
    feat = (feat * mask.unsqueeze(-1).float()).max(dim=1).values
    return feat


def get_last_token_features(feat, data):
    # feat: (batch_size, seq_len, hidden_size)
    # this is the last_hidden_state;
    # this includes padding tokens, so we need to mask them
    mask = data.attention_mask.to(feat.device)
    feat = feat[torch.arange(feat.size(0)), mask.sum(dim=1) - 1]
    return feat


def get_first_token_features(feat, data):
    # feat: (batch_size, seq_len, hidden_size)
    # this is the last_hidden_state;
    feat = feat[torch.arange(feat.size(0)), 0]
    return feat


def load_features(dataset, test_word, test_idx):
    """
    Load cached features, if exists, otherwise compute and cache them.
    """
    CACHE_FPATH = os.path.join(args.cache_dir, args.data_dir.split('/')[-1], f'{args.dataset}')
    os.makedirs(CACHE_FPATH, exist_ok=True)
    CACHE_FNAME = f'{test_word}_{args.prompt_strategy}{"-" + "-".join(args.hint.split()) if args.prompt_strategy.startswith("hint") else ""}_{args.feat_extraction_strategy}_{args.model}'

    # If cache exists then just load it, otherwise compute the features
    if not args.reset_cache and os.path.exists(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin')):
        features = torch.load(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin'))
        targets = torch.load(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_targets.bin'))
    else:
        # Compute features from the last hidden state
        if args.model.startswith('t5'):
            tokenizer = get_t5_tokenizer(args.model)
            llm_feat_extractor = T5Regressor(
                kind=args.model,
                tokenizer=tokenizer
            )
        elif args.model.startswith('llama'):
            tokenizer = get_llama2_tokenizer(args.model)
            llm_feat_extractor = Llama2Regressor(
                kind=args.model,
                tokenizer=tokenizer
            )

        # Need CUDA otherwise will be so slow!
        if args.cuda:
            llm_feat_extractor.cuda()
        llm_feat_extractor.eval()
        llm_feat_extractor.freeze_params()

        # Build the textual representation based on the prompt strategy
        prompt_builder = MyPromptBuilder(kind=args.prompt_strategy, hint=args.hint)
        data_processor = TwentyQuestionsDataProcessor(prompt_builder, tokenizer)
        dataloader = data_processor.get_dataloader(dataset, shuffle=False)

        # Forward pass through the LLM, take the aggregate (over sequence dimension)
        # of the last transformer embeddings/features
        features, targets = [], []
        for data in tqdm.tqdm(dataloader):
            with torch.no_grad():
                feat = llm_feat_extractor.forward_features(data)
                if args.feat_extraction_strategy == 'average':
                    feat = get_avg_features(feat, data)
                elif args.feat_extraction_strategy == 'max':
                    feat = get_max_features(feat, data)
                elif args.feat_extraction_strategy == 'last-token':
                    feat = get_last_token_features(feat, data)
                elif args.feat_extraction_strategy == 'first-token':
                    feat = get_first_token_features(feat, data)
            features += list(feat)
            if 'labels' in data:
                targets += list(data['labels'])

        features = torch.stack(features, dim=0)
        if len(targets) > 0:
            targets = torch.stack(targets, dim=0)
            targets = targets.clamp(min=0., max=1.)  # Used 0-1 normalization for similarity scores
        else:
            targets = cosine_similarity(features, features[test_idx])
            targets = targets.clamp(min=-1., max=1.)
        features = features.cpu()
        targets = targets.cpu()

        if args.rescale_scores:
            # Rescale scores between [0, 1]
            targets = (targets - targets.min()) / (targets.max() - targets.min())
            if targets[targets < 1].max() < 0.8:
                # Rescale scores to cover the full range in [0, 1]
                targets[targets < 1] = 0.8 * (targets[targets < 1] - targets[targets < 1].min()) / (
                        targets[targets < 1].max() - targets[targets < 1].min())
        # Cache to files
        torch.save(features, os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin'))
        torch.save(targets, os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_targets.bin'))

    return features, targets


def get_surrogate(train_x, train_y, hidden_dim=50, activation=torch.nn.Tanh, n_objs=1,
                  noise_var=0.001, hess_factorization='kron'):
    # Or just use https://github.com/wiseodd/laplace-bayesopt for full BoTorch compatibility
    feature_dim = train_x.shape[-1]

    def get_net():  # needs to be a callable for LaplaceBoTorch
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, n_objs)
        )

    model = LaplaceBoTorch(
        get_net, train_x, train_y, noise_var=noise_var, hess_factorization=hess_factorization
    )
    return model


def run_bayesopt(words, features, targets, test_word, n_init_data=10, T=None, seed=17, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sort data in descending order of targets
    targets, _idxs = torch.sort(targets, descending=True)
    features = features[_idxs]
    words = [words[i] for i in _idxs]
    word2rank = {w: r for r, w in enumerate(words, 1)}

    # Shuffle the dataset
    words, features, targets = skshuffle(words, features, targets, random_state=seed)
    ground_truth_max = targets.max().item()

    # Obtain a small initial dataset for BO.
    seen_idxs, init_x, init_y, init_x_labels = None, None, None, None
    if args.init_strategy == 'random':
        _idxs = np.random.choice(np.arange(0, len(features)), size=n_init_data + 1, replace=False)
        idxs = [_i for _i in _idxs if targets[_i].item() < ground_truth_max][:n_init_data]
        init_x = features[idxs]
        init_y = targets[idxs]
        init_x_labels = [words[_i] for _i in idxs]
        seen_idxs = set(idxs)
        seen_idxs_rand = set(idxs)
    else:
        raise NotImplementedError

    # Initialize surrogate g (learn prior from the initial dataset)
    surrogate = get_surrogate(init_x, init_y)
    surrogate = surrogate.to(device)

    # Prepare for the BO loop
    bo_found, rand_found = False, False
    best_idx = init_y.argmax()
    best_y = init_y[best_idx].item()  # Current best f(x) from the initial dataset
    best_x_label = init_x_labels[best_idx]
    best_rank = word2rank[best_x_label]
    trace_best_y = [best_y]
    trace_best_x_label = [best_x_label]
    steps_to_opt = -1

    # Also track a random sampling baseline
    best_rank_rand = word2rank[best_x_label]
    trace_best_y_rand = [best_y]
    trace_best_x_label_rand = [best_x_label]
    steps_to_opt_rand = -1

    pbar = tqdm.trange(T if T is not None else len(features) - n_init_data, file=sys.stdout)
    pbar.set_description(
        f'[Best f(x="{best_x_label}") = {best_y:.3f} (rank={word2rank[best_x_label]})]'
    )

    # The BayesOpt loop --- or just use BoTorch since LaplaceBoTorch is compatible
    for t in pbar:
        if not bo_found:
            unseen_idxs = list(set(range(len(features))) - seen_idxs)
            dataloader = data_utils.DataLoader(
                data_utils.TensorDataset(features[unseen_idxs], targets[unseen_idxs]),
                batch_size=256, shuffle=False
            )

            # Make surrogate predictions over all candidates,
            # then use the predictive mean and variance to compute the acquisition function values
            acq_vals = []
            for x, y in dataloader:
                # Obtain the posterior predictive distribution p(g_t(x) | D_t)
                posterior = surrogate.posterior(x)
                f_mean, f_var = posterior.mean, posterior.variance
                # For multiobjective problems take the covariance matrix
                # i.e., f_cov = posterior.covariance_matrix

                # Compute value of the acquisition function
                acq_fn = {
                    "thompson_sampling": thompson_sampling
                }[args.acquisition_fn]
                acq_vals.append(acq_fn(f_mean, f_var))

            # Pick the candidate that maximizes the acquisition fn and update seen idxs
            acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
            _idx_best = torch.argmax(acq_vals).item()
            idx_best = unseen_idxs[_idx_best]
            seen_idxs.add(idx_best)  # Add to seen idxs
            # Observe true value of selected candidate
            new_x, new_y = features[idx_best], targets[idx_best]
            new_x_label = words[idx_best]
            if new_y.item() > best_y:
                best_y = new_y.item()
                best_x_label = new_x_label
                best_rank = word2rank[best_x_label]
                if best_y == ground_truth_max:
                    steps_to_opt = t + 1
                    bo_found = True
            trace_best_y.append(best_y)
            trace_best_x_label.append(best_x_label)

            # Update surrogate posterior with the newly acquired (x, y)
            surrogate = surrogate.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))

            pbar.set_description(
                f'[Best f(x="{best_x_label}") = {best_y:.3f} (rank={word2rank[best_x_label]}), '
                + f'curr f(x="{new_x_label}") = {new_y.item():.3f} (rank={word2rank[new_x_label]})]'
            )
        else:
            trace_best_y.append(trace_best_y[-1])
            trace_best_x_label.append(trace_best_x_label[-1])
            pbar.set_description(
                f'[Best f(x="{best_x_label}") = {best_y:.3f} (rank={word2rank[best_x_label]})]'
            )

        # Random sampling baseline
        if not rand_found:
            unseen_idxs_rand = list(set(range(len(features))) - seen_idxs_rand)
            idx_rand = np.random.choice(unseen_idxs_rand)
            seen_idxs_rand.add(idx_rand)
            new_x_rand, new_y_rand = features[idx_rand], targets[idx_rand]
            if new_y_rand.item() > trace_best_y_rand[-1]:
                trace_best_y_rand.append(new_y_rand.item())
                trace_best_x_label_rand.append(words[idx_rand])
                best_rank_rand = word2rank[words[idx_rand]]
                if trace_best_y_rand[-1] == ground_truth_max:
                    steps_to_opt_rand = t + 1
                    rand_found = True
            else:
                trace_best_y_rand.append(trace_best_y_rand[-1])
                trace_best_x_label_rand.append(trace_best_x_label_rand[-1])
        else:
            trace_best_y_rand.append(trace_best_y_rand[-1])
            trace_best_x_label_rand.append(trace_best_x_label_rand[-1])

    if best_y == ground_truth_max:
        print(f'Hidden word ("{best_x_label}") found at step {steps_to_opt}.')
    else:
        print(
            f'Hidden word ("{test_word}") not found. Best found: f(x="{best_x_label}") = {round(best_y, 3)} (rank={best_rank}).')

    return {
        "target": test_word,
        "seed": seed,
        "n_init_data": n_init_data,
        "T": T if T is not None else None,
        "opt_val": ground_truth_max,
        "prompt_strategy": args.prompt_strategy,
        "hint": args.hint,
        "feat_extraction_strategy": args.feat_extraction_strategy,
        "model": args.model,
        "results": {
            "best_found": trace_best_x_label[-1],
            "best_y": trace_best_y[-1],
            "best_rank": best_rank,
            "steps_to_opt": steps_to_opt,
            "best_found_rand": trace_best_x_label_rand[-1],
            "best_y_rand": trace_best_y_rand[-1],
            "best_rank_rand": best_rank_rand,
            "steps_to_opt_rand": steps_to_opt_rand,
            "trace_y": trace_best_y,
            "trace_x": trace_best_x_label,
            "trace_y_rand": trace_best_y_rand,
            "trace_x_rand": trace_best_x_label_rand,
        }
    }


def plot(results, aggregate=False):
    plt.clf()
    if not aggregate:
        res = results['results']
        t = np.arange(len(res['trace_y']))
        plt.axhline(results['opt_val'], color='black', linestyle='dashed')
        plt.plot(t, res['trace_y'])
        plt.plot(t, res['trace_y_rand'])
        plt.legend(["Optimal", "BO", "Random"])
        plt.xlabel(r'$t$')
        plt.ylabel(r'Objective ($\uparrow$)')
        plt.title(
            f"best_rank={res['best_rank']}, best_x={res['trace_x'][-1]}, best_y={round(res['trace_y'][-1], 4)}, steps={res['steps_to_opt']}")
        plt.savefig(
            os.path.join(out_dir, f'{results["target"]}_T-{args.T}_init-{args.n_init_data}_seed-{results["seed"]}.png'))
        print(f'Saved plot at ' +
              os.path.join(out_dir,
                           f'{results["target"]}_T-{args.T}_init-{args.n_init_data}_seed-{results["seed"]}.png'))
    else:
        res = results['results']
        t = np.arange(len(res['trace_y_mean']))
        plt.axhline(results['opt_val'], color='black', linestyle='dashed', label="Optimal")
        plt.plot(t, res['trace_y_mean'], label="BO")
        plt.fill_between(t, np.array(res['trace_y_mean']) - np.array(res['trace_y_std']),
                         np.array(res['trace_y_mean']) + np.array(res['trace_y_std']),
                         color='blue', alpha=0.2)
        plt.plot(t, res['trace_y_mean_rand'], label="Random")
        plt.fill_between(t, np.array(res['trace_y_mean_rand']) - np.array(res['trace_y_std_rand']),
                         np.array(res['trace_y_mean_rand']) + np.array(res['trace_y_std_rand']),
                         color='orange', alpha=0.2)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'Objective ($\uparrow$)')
        plt.title(
            f"avg_rank={res['avg_rank']}, avg_rank_rand={res['avg_rank_rand']}")
        plt.savefig(os.path.join(out_dir, f'agg_{results["target"]}_T-{args.T}_init-{args.n_init_data}.png'))
        print(f'Saved final plot at ' + os.path.join(out_dir,
                                                     f'agg_{results["target"]}_T-{args.T}_init-{args.n_init_data}.png'))


if __name__ == '__main__':
    # Setup
    parser = Parser()
    global args
    args = parser.parse_args()
    print("Script arguments:")
    print(args.__dict__)
    global RUN_ID
    RUN_ID = str(int(time.time()))
    global out_dir
    out_dir = os.path.join(args.out_dir, RUN_ID)

    # Load dataset and select the test word
    pd_dataset = pd.read_csv(os.path.join(args.data_dir, f'{args.dataset}.csv'))

    if 'Similarity' not in pd_dataset.columns:
        # Compute similarities and create dataset
        if args.test_idx_or_word is None:
            test_idx = np.random.randint(len(pd_dataset))
        else:
            try:
                test_idx = int(args.test_idx_or_word)
            except ValueError:
                test_idx = pd_dataset.index[pd_dataset["Words"] == args.test_idx_or_word].tolist()[0]
        test_word = pd_dataset['Words'][test_idx]
        print(f'\nHIDDEN WORD: "{test_word}"')

        # Add word representations and compute similarities
        features, targets = load_features(dataset=pd_dataset, test_word=test_word, test_idx=test_idx)
        if args.save_word_specific_dataset:
            dataset_dir = 'data/twentyquestions/datasets'
            os.makedirs(dataset_dir, exist_ok=True)
            pd.DataFrame({
                'Words': pd_dataset['Words'],
                'Similarity': targets.tolist()
            }).sort_values(by=['Similarity'], ascending=False).to_csv(os.path.join(dataset_dir,
                                                                                   f'{test_word}_{args.prompt_strategy}{"-" + "-".join(args.hint.split()) if args.prompt_strategy.startswith("hint") else ""}_{args.feat_extraction_strategy}_{args.model}.csv'),
                                                                      sep='\t', index=False)
            print(f'Saved word-specific dataset to ' + os.path.join(dataset_dir,
                                                                    f'{test_word}_{args.prompt_strategy}{"-" + "-".join(args.hint.split()) if args.prompt_strategy.startswith("hint") else ""}_{args.feat_extraction_strategy}_{args.model}.csv'))
    else:
        test_idx = 0
        test_word = pd_dataset['Words'][test_idx]
        print(f'\nHIDDEN WORD: "{test_word}"')
        features, targets = load_features(dataset=pd_dataset, test_word=test_word, test_idx=test_idx)


    if args.exit_after_feat_extraction:
        print('\nExiting after feature extraction.')
        exit(0)

    # Run BO over multiple seeds
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    start_time = time.time()
    for i in range(args.n_seeds):
        seed = args.seed + i
        print(f'\nSeed {seed}:')
        results = run_bayesopt(words=list(pd_dataset['Words']), features=features, targets=targets,
                               test_word=test_word, n_init_data=args.n_init_data, T=args.T, seed=seed,
                               device='cuda' if args.cuda else 'cpu')
        plot(results)
        all_results.append(results)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Aggregate results
    all_trace_y = np.stack([res["results"]["trace_y"] for res in all_results])
    all_trace_y_rand = np.stack([res["results"]["trace_y_rand"] for res in all_results])
    final_res = {
        "target": all_results[-1]["target"],
        "n_seeds": args.n_seeds,
        "n_init_data": all_results[-1]["n_init_data"],
        "T": all_results[-1]["T"],
        "opt_val": all_results[-1]["opt_val"],
        "prompt_strategy": args.prompt_strategy,
        "hint": args.hint,
        "feat_extraction_strategy": args.feat_extraction_strategy,
        "model": args.model,
        "avg_elapsed_time": round(elapsed_time / args.n_seeds, 2),
        "results": {
            "avg_rank": np.mean([res["results"]["best_rank"] for res in all_results]),
            "avg_found": sum([1 for res in all_results if res["results"]["steps_to_opt"] != -1]) / args.n_seeds,
            "avg_steps_to_opt": np.mean(
                [res["results"]["steps_to_opt"] for res in all_results if res["results"]["steps_to_opt"] != -1]),
            "avg_rank_rand": np.mean([res["results"]["best_rank_rand"] for res in all_results]),
            "avg_found_rand": sum(
                [1 for res in all_results if res["results"]["steps_to_opt_rand"] != -1]) / args.n_seeds,
            "avg_steps_to_opt_rand": np.mean(
                [res["results"]["steps_to_opt_rand"] for res in all_results if
                 res["results"]["steps_to_opt_rand"] != -1]),
            "trace_y_mean": list(all_trace_y.mean(axis=0)),
            "trace_y_std": list(all_trace_y.std(axis=0)),
            "trace_y_mean_rand": list(all_trace_y_rand.mean(axis=0)),
            "trace_y_std_rand": list(all_trace_y_rand.std(axis=0)),
            "per_seed": all_results
        }
    }
    for k, v in final_res["results"].items():
        if type(v) is float:
            final_res["results"][k] = round(v, 2)

    with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
        fh.write(json.dumps(final_res, indent=2))
    print(f'\nSaved results to ' + os.path.join(out_dir, 'results.json'))
    # Plot aggregated results
    plot(final_res, aggregate=True)
