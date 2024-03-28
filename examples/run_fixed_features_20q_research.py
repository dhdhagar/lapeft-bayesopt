import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm
from sklearn.utils import shuffle as skshuffle
import os
import matplotlib.pyplot as plt

# Non-finetuned surrogates. The finetuned surrogates are in lapeft_bayesopt.surrogates
from fixed_feat_surrogate import LaplaceBoTorch

from lapeft_bayesopt.foundation_models.llama2 import Llama2Regressor
from lapeft_bayesopt.foundation_models.utils import get_llama2_tokenizer
from lapeft_bayesopt.utils.acqf import thompson_sampling
from lapeft_bayesopt.utils import helpers

# Our self-defined problems, using the format provided by lapeft-bayesopt
from data_processor import TQuestResearchDataProcessor
from prompting import MyPromptBuilder

import argparse

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument(
            "--dataset", type=str, default='research'
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
            "--interactive", action="store_true",
        )

def main():
    parser = Parser()
    global args
    args = parser.parse_args()
    print("Script arguments:")
    print(args.__dict__)

    pd_dataset = pd.read_csv(f'examples/data/{args.dataset}.csv')
    dataset = {
        'pd_dataset': pd_dataset,
        'maximization': True,
        'cache_path': f'examples/data/cache/{args.dataset}/',
        'opt_val': pd_dataset['Similarity'].max()
    }
    os.makedirs(dataset['cache_path'], exist_ok=True)

    results = run_bayesopt(dataset, n_init_data=args.n_init_data, T=args.T, randseed=args.seed)

    # Plot
    t = np.arange(len(results))
    plt.axhline(dataset['opt_val'], color='black', linestyle='dashed')
    plt.plot(t, results)
    plt.xlabel(r'$t$')
    plt.ylabel(r'Objective ($\uparrow$)')
    plt.show()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{args.dataset}_T-{args.T}_init-{args.n_init_data}_rand-{args.seed}.png')
    print(f'Saved plot at outputs/{args.dataset}_T-{args.T}_init-{args.n_init_data}_rand-{args.seed}.png')


def load_features(dataset):
    """
    Load cached features (e.g. the fingerprints of each molecules in the dataset) if exist.
    Otherwise, compute the said features and cache them.
    Also, transform the problem into maximization.
    """
    pd_dataset = dataset['pd_dataset']
    CACHE_PATH = dataset['cache_path']
    MAXIMIZATION = dataset['maximization']

    # If cache exists then just load it, otherwise compute the features
    if os.path.exists(f'{CACHE_PATH}/cached_feats.bin'):
        features = torch.load(f'{CACHE_PATH}/cached_feats.bin')
        targets = torch.load(f'{CACHE_PATH}/cached_targets.bin')
    else:
        # Use the chemistry-specific T5 LLM of Christofidellis et al., 2023
        tokenizer = get_llama2_tokenizer('llama-2-7b')
        llm_feat_extractor = Llama2Regressor(
            kind='llama-2-7b',
            tokenizer=tokenizer
        )

        # Need CUDA otherwise will be so slow!
        # llm_feat_extractor.cuda()
        llm_feat_extractor.eval()
        llm_feat_extractor.freeze_params()

        # Here, we use the raw SMILES string as the input to the LLM
        prompt_builder = MyPromptBuilder(kind='just-smiles')
        data_processor = TQuestResearchDataProcessor(prompt_builder, tokenizer)
        dataloader = data_processor.get_dataloader(pd_dataset, shuffle=False, token_id_only=True)

        # Forward pass through the LLM, take the aggregate (over sequence dimension)
        # of the last transformer embeddings/features
        features, targets = [], []
        for data in tqdm.tqdm(dataloader):
            with torch.no_grad():
                feat = llm_feat_extractor.forward_features(data, embeddings=True)

            features += list(feat.cpu())

            # Here we transform the target so that the optimization problem
            # always corresponds to maximization
            targets += list(helpers.y_transform(data['labels'], MAXIMIZATION))

        # Cache to files
        torch.save(features, f'{CACHE_PATH}/cached_feats.bin')
        torch.save(targets, f'{CACHE_PATH}/cached_targets.bin')

    return features, targets


def run_bayesopt(dataset, n_init_data=5, T=26, device='cpu', randseed=1):
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    features, targets = load_features(dataset)

    # Shuffle since some .csv datasets are ordered by the objective values
    features, targets, pd_dataset = skshuffle(features, targets, dataset['pd_dataset'], random_state=randseed)
    feature_dim = features[0].shape[-1]
    ground_truth_max = torch.tensor(targets).flatten().max()

    # Obtain a small initial dataset for BO.
    # There are different strategy for this. Here, we use a simple random sampling.
    train_x, train_y = [], []
    while len(train_x) < n_init_data:
        idx = np.random.randint(len(features))
        # Make sure that the optimum is not included
        if targets[idx].item() >= ground_truth_max:
            continue
        train_x.append(features.pop(idx))
        train_y.append(targets.pop(idx))
        pd_dataset = pd_dataset.T
        pd_pop = pd_dataset.pop(idx)
        pd_dataset = pd_dataset.T
    train_x, train_y = torch.stack(train_x), torch.stack(train_y)

    # Surrogate
    def get_net():
        activation = torch.nn.Tanh
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 50),
            activation(),
            torch.nn.Linear(50, 50),
            activation(),
            # For multiobjective problems, change 1 -> num_of_objectives
            torch.nn.Linear(50, 1)
        )
    # Or just use https://github.com/wiseodd/laplace-bayesopt for full BoTorch compatibility
    model = LaplaceBoTorch(
        get_net, train_x, train_y, noise_var=0.001, hess_factorization='kron'
    )
    model = model.to(device) if model is not None else model

    # Prepare for the BO loop
    MAXIMIZATION = dataset['maximization']
    best_y = train_y.max().item() # Current best f(x) from the initial dataset
    best_x = None
    pbar = tqdm.trange(T)
    pbar.set_description(
        f'[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}]'
    )

    # To store the logged best f(x) over time
    trace_best_y = [helpers.y_transform(best_y, MAXIMIZATION)]

    # The BayesOpt loop --- or just use BoTorch since LaplaceBoTorch is compatible
    for t in pbar:
        # Don't shuffle so that the ordering is the same as the pandas dataset
        dataloader = data_utils.DataLoader(
            data_utils.TensorDataset(torch.stack(features), torch.stack(targets)),
            batch_size=256, shuffle=False
        )

        # Make prediction over all unknown molecules, then use the predictive mean
        # and variance to compute the acquisition function values
        acq_vals = []
        for x, y in dataloader:
            # Obtain the posterior predictive distribution p(g_t(x) | D_t)
            posterior = model.posterior(x)

            # For multiobjective problems take the covariance matrix
            # i.e., f_cov = posterior.covariance_matrix
            f_mean, f_var = posterior.mean, posterior.variance

            # Feel free to use different acquisition function
            acq_vals.append(thompson_sampling(f_mean, f_var))

        # Pick a molecule (a row in the current dataset) that maximizes the acquisition.
        # Also remove it from the pool (hence we use .pop)
        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
        idx_best = torch.argmax(acq_vals).item()
        new_x, new_y = features.pop(idx_best), targets.pop(idx_best)
        pd_dataset = pd_dataset.T
        pd_pop = pd_dataset.pop(idx_best)
        pd_dataset = pd_dataset.T

        # Update the current best y
        if new_y.item() > best_y:
            best_y = new_y.item()
            best_x = pd_pop["Word"]

        # Remember that the cached features are always in maximization format.
        # So here, we transform it back if necessary.
        pbar.set_description(
            f'[Best f(x="{best_x}") = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, '
            + f'curr f(x="{pd_pop["Word"]}") = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}]'
        )

        # Concatenate the newly acquired (x, y) and then update the surrogate
        model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))

        # Log the current best f(x) value
        trace_best_y.append(helpers.y_transform(best_y, MAXIMIZATION))

    return trace_best_y


if __name__ == '__main__':
    main()
