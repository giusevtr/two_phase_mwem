import os, sys
from Util.qm import QueryManager
import argparse
import numpy as np
import time
from datasets.dataset import Dataset
import pandas as pd
from Util import oracle, util2, benchmarks
from tqdm import tqdm
from algorithms import fem, dualquery, mwem


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    args = parser.parse_args()

    print("=============================================")
    print(vars(args))
    total_runs = 5
    epsilon_list = np.linspace(0.1, 0.5, 5)

    ######################################################
    ## Get dataset
    ######################################################
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]
    delta = 1.0/N**2
    ######################################################
    ## Get Queries
    ######################################################
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    real_ans = query_manager.get_answer(data)

    # Store synthetic support
    file_prefix = f'{args.dataset[0]}_{args.workload[0]}_{args.marginal[0]}'
    support_path = 'private_support/{}_{}_{}'.format(args.dataset[0], args.workload[0], args.marginal[0])
    os.makedirs('private_support', exist_ok=True)
    os.makedirs(support_path, exist_ok=True)

    support_error = {}
    mwem_error = {}
    mwem_epsilon_values = [0.5, 1]
    RESULTS = []
    for run_id in range(total_runs):
        ith_support_path = f'{support_path}/r_{run_id}'
        os.makedirs(ith_support_path, exist_ok=True)
        np.random.seed(run_id)

        support_error[run_id] = []
        mwem_error[run_id] = {}
        for mwem_eps in mwem_epsilon_values:
            mwem_error[run_id][mwem_eps] = []

        for e in epsilon_list:
            # read FEM support
            for support_algo in ['fem', 'dualquery']:
                data_path = f'{ith_support_path}/{support_algo}_{e:.1f}.csv'
                rho_path = f'{ith_support_path}/{support_algo}_{e:.1f}.txt'
                rho_file = open(rho_path, 'r')
                cumulative_rho = float(rho_file.readline())
                rho_file.close()

                print(f'reading path = {data_path}, rho = {cumulative_rho}')

                support_dataset = Dataset(pd.read_csv(data_path), domain=data.domain)
                fd_sz = support_dataset.df.shape[0]
                support_max_error = np.abs(real_ans - query_manager.get_answer(support_dataset)).max()

                for mwem_eps in mwem_epsilon_values:
                    mwem_support, A = mwem.generate(support_dataset, real_ans, N, query_manager,
                                                    epsilon=mwem_eps, delta=delta,
                                                    epsilon_split=0.02,
                                                    cumulative_rho=cumulative_rho)
                    mwem_max_error = np.abs(real_ans - query_manager.get_answer(mwem_support, A)).max()

                    # run BO to find the best error
                    _, best_epsilon_split, best_error = mwem.bo_search(support_dataset, real_ans, N, query_manager,
                                                    epsilon=mwem_eps, delta=delta,
                                                    epsilon_split=0.02,
                                                    cumulative_rho=cumulative_rho,
                                                    epslon_split_range=[0.01, 0.05])

                    row = [run_id, support_algo, e, mwem_eps, support_max_error, mwem_max_error,  best_error, best_epsilon_split]
                    RESULTS.append(row)
                    print(f'{row}')
    # return support_error, mwem_error


    results_dir = 'ResultsMWEM'
    os.makedirs(results_dir, exist_ok=True)
    results_path = f'{results_dir}/{file_prefix}.csv'
    print(f'saving results in {results_path}')
    columns = ['run_id', 'support_algorithm', 'support_epsilon', 'total_epsilon', 'support_error', 'mwem_error', 'best_mwem_error', 'best_epsilon_split']

    df = pd.DataFrame(RESULTS, columns=columns)
    df.to_csv(results_path, index=False)
