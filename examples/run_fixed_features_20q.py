import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm
from sklearn.utils import shuffle as skshuffle
import os
import time
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

import argparse


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--data_dir", type=str, default='examples/data'
        )
        self.add_argument(
            "--dataset", type=str, default='twentyquestions-dev-1000'
        )
        self.add_argument(
            "--test_idx_or_word", type=str
        )
        self.add_argument(
            "--prompt_type", type=str, choices=['word', 'completion', 'instruction'], default='word'
        )
        self.add_argument(
            "--model", type=str, choices=['t5-small', 't5-base', 'llama-2-7b', 'llama-2-13b'], default="llama2-7b"
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
            "--rescale_scores", action=argparse.BooleanOptionalAction, default=True
        )
        self.add_argument(
            "--seed", type=int, default=9999
        )
        self.add_argument(
            "--n_init_data", type=int, default=2
        )
        self.add_argument(
            "--T", type=int, default=20
        )
        self.add_argument(
            "--n_seeds", type=int, default=1
        )
        self.add_argument(
            "--debug", action=argparse.BooleanOptionalAction, default=False
        )


def load_features(dataset, test_word, test_idx, prompt_type):
    """
    Load cached features, if exists, otherwise compute and cache them.
    """
    CACHE_FPATH = os.path.join(args.data_dir, f'cache/{args.dataset}/')
    os.makedirs(CACHE_FPATH, exist_ok=True)
    CACHE_FNAME = f'{test_word}_{prompt_type}'

    # If cache exists then just load it, otherwise compute the features
    if os.path.exists(os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin')):
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
        prompt_builder = MyPromptBuilder(kind=prompt_type)
        data_processor = TwentyQuestionsDataProcessor(prompt_builder, tokenizer)
        dataloader = data_processor.get_dataloader(dataset, shuffle=False, token_id_only=True)

        # Forward pass through the LLM, take the aggregate (over sequence dimension)
        # of the last transformer embeddings/features
        features, targets = [], []
        for data in tqdm.tqdm(dataloader):
            with torch.no_grad():
                feat = llm_feat_extractor.forward_features(data, embeddings=True)

            features += list(feat)

            # # Here we transform the target so that the optimization problem
            # # always corresponds to maximization
            # targets += list(helpers.y_transform(data['labels'], MAXIMIZATION))
        features = torch.cat(features, dim=0)
        targets = cosine_similarity(features, features[None, test_idx]).cpu()
        assert targets.max() == 1
        if args.rescale_scores:
            # Rescale scores between [0, 1]
            targets = (targets - targets.min()) / (targets.max() - targets.min())
            if targets[targets < 1].max() < 0.9:
                # Rescale scores to cover the full range in [0, 1]
                targets[targets < 1] = 0.9 * (targets[targets < 1] - targets[targets < 1].min()) / (
                        targets[targets < 1].max() - targets[targets < 1].min())
        features = features.cpu()
        # Cache to files
        torch.save(features, os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_feats.bin'))
        torch.save(targets, os.path.join(CACHE_FPATH, f'{CACHE_FNAME}_targets.bin'))

    return features, targets


def get_surrogate(train_x, train_y, hidden_dim=50, activation=torch.nn.Tanh, n_objs=1,
                  noise_var=0.001, hess_factorization='kron'):
    # Or just use https://github.com/wiseodd/laplace-bayesopt for full BoTorch compatibility
    feature_dim = train_x.shape[-1]
    net = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, hidden_dim),
        activation(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        activation(),
        torch.nn.Linear(hidden_dim, n_objs)
    )
    model = LaplaceBoTorch(
        net, train_x, train_y, noise_var=noise_var, hess_factorization=hess_factorization
    )
    return model


def run_bayesopt(words, features, targets, test_word, n_init_data=10, T=None, seed=17, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Shuffle the dataset
    words, features, targets = skshuffle(words, features, targets, random_state=seed)
    ground_truth_max = max(targets)

    # Obtain a small initial dataset for BO.
    seen_idxs, init_x, init_y, init_x_labels = None, None, None, None
    if args.init_strategy == 'random':
        _idxs = np.random.choice(np.arange(0, len(features)), size=n_init_data + 1, replace=False)
        idxs = [_i for _i in _idxs if targets[_i].item() < ground_truth_max][:n_init_data]
        init_x = torch.stack(features[idxs])
        init_y = torch.stack(targets[idxs])
        init_x_labels = [words[_i] for _i in idxs]
        seen_idxs = set(idxs)
    else:
        raise NotImplementedError

    # Initialize surrogate g (learn prior from the initial dataset)
    surrogate = get_surrogate(init_x, init_y)
    surrogate = surrogate.to(device)

    # Prepare for the BO loop
    best_idx = init_y.argmax()
    best_y = init_y[best_idx].item()  # Current best f(x) from the initial dataset
    best_x_label = init_x_labels[best_idx]
    trace_best_y = [best_y]
    trace_best_x_label = [best_x_label]
    steps_to_opt = -1

    # Also track a random sampling baseline
    trace_best_y_rand = [best_y]
    trace_best_x_label_rand = [best_x_label]
    steps_to_opt_rand = -1

    pbar = tqdm.trange(T if T is not None else len(features) - n_init_data)
    pbar.set_description(
        f'[Best f(x="{best_x_label}") = {best_y:.4f}]'
    )

    # The BayesOpt loop --- or just use BoTorch since LaplaceBoTorch is compatible
    for t in pbar:
        unseen_idxs = list(set(range(len(features))) - seen_idxs)
        dataloader = data_utils.DataLoader(
            data_utils.TensorDataset(torch.stack(features[unseen_idxs]), torch.stack(targets[unseen_idxs])),
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
        idx_best = torch.argmax(acq_vals).item()
        seen_idxs.add(unseen_idxs[idx_best])  # Add to seen idxs
        # Observe true value of selected candidate
        new_x, new_y = features[idx_best], targets[idx_best]
        new_x_label = words[unseen_idxs[idx_best]]
        if new_y.item() > best_y:
            best_y = new_y.item()
            best_x_label = new_x_label
            if best_y == ground_truth_max:
                steps_to_opt = t + 1
        trace_best_y.append(best_y)
        trace_best_x_label.append(best_x_label)

        # Update surrogate posterior with the newly acquired (x, y)
        surrogate = surrogate.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))

        pbar.set_description(
            f'[Best f(x="{best_x_label}") = {best_y:.3f}, '
            + f'curr f(x="{new_x_label}") = {new_y.item():.3f}]'
        )

        # Random sampling baseline
        idx_rand = np.random.randint(len(features))
        new_x_rand, new_y_rand = features[idx_rand], targets[idx_rand]
        if new_y_rand.item() > trace_best_y_rand[-1]:
            trace_best_y_rand.append(new_y_rand.item())
            trace_best_x_label_rand.append(words[idx_rand])
            if trace_best_y_rand[-1] == ground_truth_max:
                steps_to_opt_rand = t + 1
        else:
            trace_best_y_rand.append(trace_best_y_rand[-1])

    if best_y == ground_truth_max:
        print(f'Optimum "{best_x_label}" found at step {steps_to_opt}.')
    else:
        print(f'Optimum "{test_word}" not found. Best found: f(x="{best_x_label}") = {round(best_y, 3)}.')

    return {
        "target": test_word,
        "seed": seed,
        "n_init_data": n_init_data,
        "T": T if T is not None else None,
        "opt_val": ground_truth_max,
        "results": {
            "trace_y": trace_best_y,
            "trace_x": trace_best_x_label,
            "steps_to_opt": steps_to_opt,
            "trace_y_rand": trace_best_y_rand,
            "trace_x_rand": trace_best_x_label_rand,
            "steps_to_opt_rand": steps_to_opt_rand,
        }
    }


def plot(results):
    res = results['results']
    t = np.arange(len(results['trace_y']))
    plt.clf()
    plt.axhline(results['opt_val'], color='black', linestyle='dashed')
    plt.plot(t, res['trace_y'])
    plt.plot(t, res['trace_y_rand'])
    plt.legend(["Optimal", "BO", "Random"])
    plt.xlabel(r'$t$')
    plt.ylabel(r'Objective ($\uparrow$)')
    plt.title(f"steps={res['steps_to_opt']}, best_x={res['trace_x'][-1]}, best_y={res['trace_y'][-1]}")
    plt.savefig(
        os.path.join(out_dir, f'{results["target"]}_T-{args.T}_init-{args.n_init_data}_seed-{seed}.png'))
    print(f'Saved plot at ' +
          os.path.join(out_dir, f'{results["target"]}_T-{args.T}_init-{args.n_init_data}_seed-{seed}.png'))


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
    out_dir = os.path.join("outputs", RUN_ID)
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset and select the test word
    pd_dataset = pd.read_csv(os.path.join(args.data_dir, f'{args.dataset}.csv'))
    if args.test_idx_or_word is None:
        test_idx = np.random.randint(len(pd_dataset))
    else:
        try:
            test_idx = int(args.test_idx_or_word)
        except ValueError:
            test_idx = pd_dataset.index[pd_dataset["Words"] == args.test_idx_or_word].tolist()[0]
    test_word = pd_dataset['Words'][test_idx]
    print(f"\nTest word: {test_word}\n")

    # Add word representations and compute similarities
    features, targets = load_features(dataset=pd_dataset, test_word=test_word,
                                      test_idx=test_idx, prompt_type=args.prompt_type)

    # Run BO over multiple seeds
    all_results = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        results = run_bayesopt(words=list(pd_dataset['Words']), features=features, targets=targets,
                               test_word=test_word, n_init_data=args.n_init_data, T=args.T, randseed=seed)
        plot(results)
        all_results.append(results)

    # Aggregate results
    all_trace_y = np.stack([res["results"]["trace_y"] for res in all_results])
    all_trace_y_rand = np.stack([res["results"]["trace_y_rand"] for res in all_results])
    final_res = {
        "target": all_results[-1]["target"],
        "n_seeds": args.n_seeds,
        "n_init_data": all_results[-1]["n_init_data"],
        "T": all_results[-1]["T"],
        "opt_val": all_results[-1]["opt_val"],
        "results": {
            "trace_y_mean": all_trace_y.mean(axis=0),
            "trace_y_std": all_trace_y.std(axis=0),
            "n_found": [1 for res in all_results if res["results"]["steps_to_opt"] != -1]/args.n_seeds,
            "avg_steps_to_opt": np.mean(
                [res["results"]["steps_to_opt"] for res in all_results if res["results"]["steps_to_opt"] != -1]),
            "trace_y_rand_mean": all_trace_y_rand.mean(axis=0),
            "trace_y_rand_std": all_trace_y_rand.std(axis=0),
            "n_found_rand": [1 for res in all_results if res["results"]["steps_to_opt_rand"] != -1]/args.n_seeds,
            "avg_steps_to_opt_rand": np.mean(
                [res["results"]["steps_to_opt_rand"] for res in all_results if
                 res["results"]["steps_to_opt_rand"] != -1]),
            "per_seed": all_results
        },
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
        fh.write(json.dumps(final_res, indent=2))
