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
import matplotlib.animation as animation
from torch.nn.functional import cosine_similarity

# Non-finetuned surrogates. The finetuned surrogates are in lapeft_bayesopt.surrogates
# from fixed_feat_surrogate import LaplaceBoTorch
from laplace_bayesopt.botorch import LaplaceBoTorch
from lapeft_bayesopt.foundation_models.t5 import T5Regressor
from lapeft_bayesopt.foundation_models.llama2 import Llama2Regressor
from lapeft_bayesopt.foundation_models.utils import get_llama2_tokenizer, get_t5_tokenizer
from lapeft_bayesopt.utils.acqf import ThompsonSampling  # , thompson_sampling, ucb, ei
from lapeft_bayesopt.utils import helpers
# Our self-defined problems, using the format provided by lapeft-bayesopt
from data_processor import TwentyQuestionsDataProcessor
from prompting import MyPromptBuilder
from soft_prompt import get_virtual_token

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--run_id", type=str
        )
        self.add_argument(
            "--data_dir", type=str, default='data/twentyquestions/datasets/word2vec-1000'
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
            "--dtype", type=str, choices=['fp16', 'fp32'], default=None
        )
        self.add_argument(
            "--init_strategy", type=str, choices=['random'], default="random"
        )
        self.add_argument(
            "--surrogate_fn", type=str, choices=['laplace', 'gp'], default="gp"
        )
        self.add_argument(
            "--acquisition_fn", type=str, choices=['thompson_sampling', 'logEI', 'random'], default="logEI"
        )  # 'ucb', 'ei',
        self.add_argument(
            "--optimize_acq", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--optimize_acq_unbounded", action=argparse.BooleanOptionalAction, default=False
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
            "--feat_extraction_strategy", type=str, choices=['last-token', 'average', 'max', 'first-token', 'vtoken'],
            default="last-token"
        )
        self.add_argument(
            "--additive_features", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--exit_after_feat_extraction", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--visualize_posterior", action=argparse.BooleanOptionalAction, default=False
        )
        self.add_argument(
            "--visualize_posterior_anim", action=argparse.BooleanOptionalAction, default=True
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
            "--plot_y", type=str, choices=['obj', 'rank'], default='rank'
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
    CACHE_FNAME = [
        test_word,
        f'{args.prompt_strategy}{"-" + "-".join(args.hint.split()) if args.prompt_strategy.startswith("hint") else ""}',
        args.feat_extraction_strategy,
        args.model
    ]
    is_vtoken_feat_extraction = args.feat_extraction_strategy == 'vtoken'
    if args.additive_features:
        CACHE_FNAME.append('additive')
        assert not is_vtoken_feat_extraction
    CACHE_FNAME = '_'.join(CACHE_FNAME)
    # If cache exists then just load it, otherwise compute the features
    if not args.reset_cache and os.path.exists(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin')):
        features = torch.load(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin'))
        targets = torch.load(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_targets.bin'))
        print('\nLoaded cached features.')
    else:
        dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(args.dtype, None)
        # Compute features from the last hidden state
        if args.model.startswith('t5'):
            tokenizer = get_t5_tokenizer(args.model)
            llm_feat_extractor = T5Regressor(
                kind=args.model,
                tokenizer=tokenizer,
                encoder_only=not is_vtoken_feat_extraction,
                dtype=dtype
            )
        elif args.model.startswith('llama'):
            tokenizer = get_llama2_tokenizer(args.model)
            llm_feat_extractor = Llama2Regressor(
                kind=args.model,
                tokenizer=tokenizer,
                dtype=dtype
            )

        if not is_vtoken_feat_extraction and args.cuda:
            llm_feat_extractor.cuda()
        llm_feat_extractor.eval()
        llm_feat_extractor.freeze_params()

        # Build the textual representation based on the prompt strategy
        prompt_builder = MyPromptBuilder(kind=args.prompt_strategy, hint=args.hint)
        data_processor = TwentyQuestionsDataProcessor(prompt_builder, tokenizer)
        dataloader = data_processor.get_dataloader(dataset, additive=args.additive_features,
                                                   vtoken=is_vtoken_feat_extraction)

        # Get features and target values for each candidate
        features, targets = [], []
        for data in tqdm.tqdm(dataloader):
            if is_vtoken_feat_extraction:
                # Backpropagate through the LLM to learn features as virtual tokens
                feat = get_virtual_token(llm_feat_extractor.feature_extractor, tokenizer, data,
                                         device='cuda' if args.cuda else 'cpu')
            else:
                # Forward pass through the LLM, take the aggregate (over sequence dimension)
                # of the last hidden state
                with torch.no_grad():
                    data['input_ids'] = data['input_ids'].squeeze()
                    data['attention_mask'] = data['attention_mask'].squeeze()
                    feat = llm_feat_extractor.forward_features(data)
                    if args.additive_features:
                        feat = feat.mean(dim=0).unsqueeze(dim=0)
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


def get_surrogate(train_x, train_y, n_objs=1, standardize=True, device='cpu',
                  bnn_hidden_dim=50, bnn_activation=torch.nn.Tanh, bnn_noise_var=0.001, bnn_hess_factorization='kron',
                  gp_noise=None):
    # Or just use https://github.com/wiseodd/laplace-bayesopt for full BoTorch compatibility
    feature_dim = train_x.shape[-1]

    def get_net():  # needs to be a callable for LaplaceBoTorch
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, bnn_hidden_dim),
            bnn_activation(),
            torch.nn.Linear(bnn_hidden_dim, bnn_hidden_dim),
            bnn_activation(),
            torch.nn.Linear(bnn_hidden_dim, n_objs)
        ).to(device)

    if train_y.size(-1) != 1:
        train_y = train_y.unsqueeze(-1)
    if args.surrogate_fn == "laplace":
        model = LaplaceBoTorch(
            get_net, train_x.to(device), train_y.to(device), noise_var=bnn_noise_var,
            hess_factorization=bnn_hess_factorization, outcome_transform=Standardize(m=1) if standardize else None,
            device=device
        )
    elif args.surrogate_fn == "gp":
        train_yvar = None  # learned noise
        if gp_noise == 0:  # no noise
            train_yvar = torch.full_like(train_Y, 1e-6)
        elif gp_noise is not None:  # fixed noise
            train_yvar = torch.full_like(train_Y, gp_noise)
        model = SingleTaskGP(train_x, train_y,
                             train_Yvar=train_yvar,
                             outcome_transform=Standardize(m=1) if standardize else None)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    else:
        raise NotImplementedError
    return model.to(device)


def optimize_acqf_and_get_observation(acq_fn, features, unseen_idxs, device):
    """Optimizes the acquisition function, and returns a
    new candidate and a noisy observation"""

    if acq_fn is None:
        # Randomly sample from the unseen indices
        return np.random.choice(unseen_idxs)

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_fn,
        bounds=torch.stack(
            [  # Restrict to the min-max range of the LLM activation function
                torch.ones(features.shape[1], device=device) * -1.,
                torch.ones(features.shape[1], device=device) * 1.
            ] if args.optimize_acq_unbounded else
            [  # Restrict to the min-max range of the candidate set features
                features.min(dim=0).values.to(device),
                features.max(dim=0).values.to(device),
            ]
        ),
        q=1,
        num_restarts=1,
        raw_samples=1,
    )

    # observe new values
    # new_x = unnormalize(candidates.detach(), bounds=bounds)
    candidates = candidates.detach()
    # Find idx of vector in features that is closest to the candidate
    sims = cosine_similarity(features[unseen_idxs].to(device), candidates)
    _idx_best = sims.argmax().item()
    idx_best = unseen_idxs[_idx_best]
    return idx_best


def run_bayesopt(words, features, targets, test_word, n_init_data=10, T=None, seed=17, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sort data in descending order of targets
    targets, _idxs = torch.sort(targets, descending=True)
    features = features[_idxs]
    words = [words[i] for i in _idxs]
    word2rank = {w: r for r, w in enumerate(words, 1)}

    # Shuffle the dataset
    shuffled_idxs, words, features, targets = skshuffle(np.arange(len(words)), words, features, targets,
                                                        random_state=seed)
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
    surrogate = get_surrogate(init_x, init_y, device=device)

    # Prepare for the BO loop
    bo_found, rand_found = False, False
    best_idx = init_y.argmax()
    best_y = init_y[best_idx].item()  # Current best f(x) from the initial dataset
    best_x_label = init_x_labels[best_idx]
    best_rank = word2rank[best_x_label]
    trace_best_y = [best_y]
    trace_best_rank = [best_rank]
    trace_best_x_label = [best_x_label]
    trace_y = init_y.tolist()
    trace_rank = [word2rank[w] for w in init_x_labels]
    trace_x_label = init_x_labels
    steps_to_opt = -1

    # Also track a random sampling baseline
    best_rank_rand = word2rank[best_x_label]
    trace_best_y_rand = [best_y]
    trace_best_rank_rand = [best_rank_rand]
    trace_best_x_label_rand = [best_x_label]
    steps_to_opt_rand = -1

    pbar = tqdm.trange(T if T is not None else len(features) - n_init_data, file=sys.stdout)
    pbar.set_description(
        f'[Best f(x="{best_x_label}") = {best_y:.3f} (rank={word2rank[best_x_label]})]'
    )

    # Track posterior values for visualization
    posterior_vals = {}

    # The BayesOpt loop --- or just use BoTorch since LaplaceBoTorch is compatible
    for t in pbar:
        if not bo_found:
            unseen_idxs = list(set(range(len(features))) - seen_idxs)
            acq_fn = {
                "logEI": LogExpectedImprovement(model=surrogate, best_f=best_y),
                "thompson_sampling": ThompsonSampling(model=surrogate),
                "random": None
            }[args.acquisition_fn]
            if args.optimize_acq:
                # Optimize acquisition function
                idx_best = optimize_acqf_and_get_observation(acq_fn, features, unseen_idxs, device=device)
            else:
                # Perform finite-set search
                dataloader = data_utils.DataLoader(
                    data_utils.TensorDataset(features[unseen_idxs], targets[unseen_idxs]),
                    batch_size=256, shuffle=False
                )
                # Make surrogate predictions over all candidates,
                # then use the predictive mean and variance to compute the acquisition function values
                acq_vals = []
                for x, y in dataloader:
                    with torch.no_grad():
                        acq_vals.append(acq_fn(x.to(device).unsqueeze(1)).squeeze())
                # Pick the candidate that maximizes the acquisition fn and update seen idxs
                acq_vals = torch.cat(acq_vals, dim=0).cpu()
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
            trace_best_rank.append(best_rank)
            trace_best_x_label.append(best_x_label)
            trace_y.append(new_y.item())
            trace_rank.append(word2rank[new_x_label])
            trace_x_label.append(new_x_label)

            if args.visualize_posterior:
                dataloader = data_utils.DataLoader(
                    data_utils.TensorDataset(features[np.argsort(shuffled_idxs)], targets[np.argsort(shuffled_idxs)]),
                    batch_size=256, shuffle=False
                )
                f_vals = []
                for x, y in dataloader:
                    posterior = surrogate.posterior(x.to(device))
                    with torch.no_grad():
                        f_vals.append(torch.stack(
                            (y.to(device), posterior.mean.squeeze(), posterior.variance.sqrt().squeeze()), dim=-1))
                f_vals = torch.cat(f_vals, dim=0).tolist()
                posterior_vals[t] = f_vals

            # Update surrogate posterior with the newly acquired (x, y)
            surrogate = surrogate.condition_on_observations(new_x.reshape(1, -1).to(device),
                                                            new_y.reshape(1, -1).to(device))

            pbar.set_description(
                f'[Best f(x="{best_x_label}") = {best_y:.3f} (rank={word2rank[best_x_label]}), '
                + f'curr f(x="{new_x_label}") = {new_y.item():.3f} (rank={word2rank[new_x_label]})]'
            )
        else:
            trace_best_y.append(trace_best_y[-1])
            trace_best_rank.append(trace_best_rank[-1])
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
                best_rank_rand = word2rank[words[idx_rand]]
                trace_best_y_rand.append(new_y_rand.item())
                trace_best_x_label_rand.append(words[idx_rand])
                trace_best_rank_rand.append(best_rank_rand)
                if trace_best_y_rand[-1] == ground_truth_max:
                    steps_to_opt_rand = t + 1
                    rand_found = True
            else:
                trace_best_y_rand.append(trace_best_y_rand[-1])
                trace_best_rank_rand.append(trace_best_rank_rand[-1])
                trace_best_x_label_rand.append(trace_best_x_label_rand[-1])
        else:
            trace_best_y_rand.append(trace_best_y_rand[-1])
            trace_best_rank_rand.append(trace_best_rank_rand[-1])
            trace_best_x_label_rand.append(trace_best_x_label_rand[-1])

    if best_y == ground_truth_max:
        print(f'Hidden word ("{best_x_label}") found at step {steps_to_opt}.')
    else:
        print(
            f'Hidden word ("{test_word}") not found. Best found: f(x="{best_x_label}") = {round(best_y, 3)} (rank={best_rank}).')
    print(
        f'Best found by random search: f(x="{trace_best_x_label_rand[-1]}") = {round(trace_best_y_rand[-1], 3)} (rank={trace_best_rank_rand[-1]}).')

    if args.visualize_posterior:
        posterior_path = os.path.join(out_dir, f'posterior_seed{seed}.json')
        plot_posterior(posterior_vals=posterior_vals, obs_y=trace_y, obs_y_rank=trace_rank, n_warm_start=n_init_data,
                       animate=args.visualize_posterior_anim, anim_interval=300, anim_repeat=True, hide_optim=False,
                       path=posterior_path)
        with open(posterior_path, 'w') as fh:
            fh.write(json.dumps(posterior_vals, indent=2))

    return {
        "target": test_word,
        "seed": seed,
        "n_init_data": n_init_data,
        "T": T if T is not None else None,
        "opt_val": ground_truth_max,
        "prompt_strategy": args.prompt_strategy,
        "hint": args.hint,
        "feat_extraction_strategy": args.feat_extraction_strategy,
        "additive_features": args.additive_features,
        "model": args.model,
        "surrogate_fn": args.surrogate_fn,
        "acquisition_fn": args.acquisition_fn,
        "acq_optimization": "evaluate" if not args.optimize_acq \
            else f"optimize_{'unbounded' if args.optimize_acq_unbounded else 'bounded'}",
        "results": {
            "best_found": trace_best_x_label[-1],
            "best_y": trace_best_y[-1],
            "best_rank": best_rank,
            "steps_to_opt": steps_to_opt,
            "best_found_rand": trace_best_x_label_rand[-1],
            "best_y_rand": trace_best_y_rand[-1],
            "best_rank_rand": best_rank_rand,
            "steps_to_opt_rand": steps_to_opt_rand,
            "trace_best_y": trace_best_y,
            "trace_best_rank": trace_best_rank,
            "trace_best_x": trace_best_x_label,
            "trace_y": trace_y,
            "trace_rank": trace_rank,
            "trace_x": trace_x_label,
            "trace_best_y_rand": trace_best_y_rand,
            "trace_best_rank_rand": trace_best_rank_rand,
            "trace_best_x_rand": trace_best_x_label_rand,
        }
    }


def plot(results, aggregate=False):
    plt.clf()
    if not aggregate:
        res = results['results']
        if args.plot_y == 'obj':
            y_key = 'trace_best_y'
            opt_val = results['opt_val']
            y_label = r'Objective ($\uparrow$)'
        elif args.plot_y == 'rank':
            y_key = 'trace_best_rank'
            opt_val = 1
            y_label = r'Rank ($\downarrow$)'
        title_experiment = f"word={results['target']}, model={results['model']}, feat={results['feat_extraction_strategy']}, prompt={results['prompt_strategy']}{(', hint=' + results['hint']) if results['prompt_strategy'].startswith('hint') else ''}, additive={results['additive_features']}, n_init_data={results['n_init_data']}, seed={results['seed']}"
        title_result = f"best\_x={res['trace_x'][-1]}, best\_rank={res['best_rank']}, best\_obj={round(res['trace_y'][-1], 4)}, steps={res['steps_to_opt']}"
        t = np.arange(len(res[y_key]))
        plt.axhline(opt_val, color='black', linestyle='dashed', label="Optimal")
        plt.plot(t, res[y_key], label="BO")
        plt.plot(t, res[f'{y_key}_rand'], label="Random")
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(y_label)
        plt.title(f'{title_experiment}\n' + fr'$\bf{{{title_result}}}$', wrap=True)
        plt.grid()
        plt.savefig(os.path.join(out_dir, f'seed-{results["seed"]}.png'), bbox_inches="tight")
        print(f'Saved plot at ' + os.path.join(out_dir, f'seed-{results["seed"]}.png'))
    else:
        res = results['results']
        if args.plot_y == 'obj':
            y_key = 'trace_best_y'
            opt_val = results['opt_val']
            y_label = r'Objective ($\uparrow$)'
            title_result = f"avg\_obj={res['trace_y_mean'][-1]} avg\_obj\_rand={res['trace_y_mean_rand'][-1]}"
        elif args.plot_y == 'rank':
            y_key = 'trace_best_rank'
            opt_val = 1
            y_label = r'Rank ($\downarrow$)'
            title_result = f"avg\_rank={res['avg_rank']}, avg\_rank\_rand={res['avg_rank_rand']}"
        title_experiment = f"word={results['target']}, model={results['model']}, feat={results['feat_extraction_strategy']}, prompt={results['prompt_strategy']}{(', hint=' + results['hint']) if results['prompt_strategy'].startswith('hint') else ''}, additive={results['additive_features']}, n_init_data={results['n_init_data']}, n_seeds={results['n_seeds']}"
        t = np.arange(len(res[f'{y_key}_mean']))
        plt.axhline(opt_val, color='black', linestyle='dashed', label="Optimal")
        plt.plot(t, res[f'{y_key}_mean'], label="BO")
        plt.fill_between(t, np.array(res[f'{y_key}_mean']) - np.array(res[f'{y_key}_std']),
                         np.array(res[f'{y_key}_mean']) + np.array(res[f'{y_key}_std']),
                         color='blue', alpha=0.2)
        plt.plot(t, res[f'{y_key}_mean_rand'], label="Random")
        plt.fill_between(t, np.array(res[f'{y_key}_mean_rand']) - np.array(res[f'{y_key}_std_rand']),
                         np.array(res[f'{y_key}_mean_rand']) + np.array(res[f'{y_key}_std_rand']),
                         color='orange', alpha=0.2)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(y_label)
        plt.title(f'{title_experiment}\n' + fr'$\bf{{{title_result}}}$', wrap=True)
        plt.grid()
        plt.savefig(os.path.join(out_dir, f'aggregate.png'), bbox_inches="tight")
        print(f'Saved final plot at ' + os.path.join(out_dir, f'aggregate.png'))


def plot_posterior(posterior_vals, obs_y, obs_y_rank, n_warm_start, path, animate=False, anim_interval=300,
                   anim_repeat=True, hide_optim=False):
    all_vals = np.array([posterior_vals[k] for k in posterior_vals])
    plt.clf()
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        vals = all_vals[len(all_vals) - 1 if not animate else (frame if frame < len(all_vals) else (frame - 1))]
        vals = vals[(1 if hide_optim else 0):, :]
        y, mean, std = vals[:, 0], vals[:, 1], vals[:, 2]
        x = 1 + np.arange(len(y))
        ax.plot(x, y, label="True")
        ax.plot(x, mean, label="Posterior", color="orange", alpha=0.8)
        ax.scatter(obs_y_rank[:n_warm_start], obs_y[:n_warm_start], label="Warm-start", color="gray")
        ax.scatter(obs_y_rank[n_warm_start + 1:n_warm_start + frame], obs_y[n_warm_start + 1:n_warm_start + frame],
                   label="BO Observation", color="red")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="orange")
        posterior_trend = np.polyfit(x, mean, 8)
        posterior_trend = np.poly1d(posterior_trend)
        posterior_trend_y = posterior_trend(x)
        ax.plot(x, posterior_trend_y, "r--", label=f"Posterior Trend")  # (s={_posterior_trend[0]})
        ax.legend()
        ax.set_title(f"t = {frame}")
        ax.set_ylabel("Objective")
        # Set y limit to be the same for all frames
        ax.set_ylim([-5, 5] if args.surrogate_fn == "laplace" else [-0.1, 1.1])
        ax.set_xlabel("Rank")
        ax.grid()

    update(frame=0)  # initial plot

    if animate:
        ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(all_vals) + 1), interval=anim_interval,
                                      repeat=anim_repeat, repeat_delay=50)
        ani.save(path.replace(".json", ".gif"), writer="pillow")  # imagemagick
    else:
        plt.savefig(path.replace(".json", ".png"), bbox_inches="tight")


if __name__ == '__main__':
    # Setup
    parser = Parser()
    global args
    args = parser.parse_args()
    print("Script arguments:")
    print(args.__dict__)
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else args.run_id
    global out_dir
    out_dir = os.path.join(args.out_dir, RUN_ID)
    print(f'Output directory: {out_dir}')

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
        test_idx = 0  # assuming dataset is sorted by decreasing similarity
        test_word = pd_dataset['Words'][test_idx]
        print(f'\nHIDDEN WORD: "{test_word}"')
        features, targets = load_features(dataset=pd_dataset, test_word=test_word, test_idx=test_idx)
        # TEMP FIX; TODO: Fix shape of saved features
        features = features.squeeze()
        targets = targets.squeeze()

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
    all_trace_y = np.stack([res["results"]["trace_best_y"] for res in all_results])
    all_trace_rank = np.stack([res["results"]["trace_best_rank"] for res in all_results])
    all_trace_y_rand = np.stack([res["results"]["trace_best_y_rand"] for res in all_results])
    all_trace_rank_rand = np.stack([res["results"]["trace_best_rank_rand"] for res in all_results])
    final_res = {
        "target": all_results[-1]["target"],
        "n_seeds": args.n_seeds,
        "n_init_data": all_results[-1]["n_init_data"],
        "T": all_results[-1]["T"],
        "opt_val": all_results[-1]["opt_val"],
        "prompt_strategy": args.prompt_strategy,
        "hint": args.hint,
        "feat_extraction_strategy": args.feat_extraction_strategy,
        "additive_features": args.additive_features,
        "model": args.model,
        "surrogate_fn": args.surrogate_fn,
        "acquisition_fn": args.acquisition_fn,
        "acq_optimization": "evaluate" if not args.optimize_acq \
            else f"optimize_{'unbounded' if args.optimize_acq_unbounded else 'bounded'}",
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
            "trace_best_y_mean": list(all_trace_y.mean(axis=0)),
            "trace_best_y_std": list(all_trace_y.std(axis=0)),
            "trace_best_rank_mean": list(all_trace_rank.mean(axis=0)),
            "trace_best_rank_std": list(all_trace_rank.std(axis=0)),
            "trace_best_y_mean_rand": list(all_trace_y_rand.mean(axis=0)),
            "trace_best_y_std_rand": list(all_trace_y_rand.std(axis=0)),
            "trace_best_rank_mean_rand": list(all_trace_rank_rand.mean(axis=0)),
            "trace_best_rank_std_rand": list(all_trace_rank_rand.std(axis=0)),
            "per_seed": all_results
        }
    }
    for k, v in final_res["results"].items():
        if type(v) is float:
            final_res["results"][k] = round(v, 2)
        if k in ["avg_steps_to_opt", "avg_steps_to_opt_rand"]:
            final_res["results"][k] = None if np.isnan(v) else v

    with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
        fh.write(json.dumps(final_res, indent=2))
    print(f'\nSaved results to ' + os.path.join(out_dir, 'results.json'))
    # Plot aggregated results
    plot(final_res, aggregate=True)
