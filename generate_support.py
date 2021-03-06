import os, sys
from Util.qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
from Util import oracle, util2, benchmarks
from tqdm import tqdm
from algorithms import fem, dualquery


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    args = parser.parse_args()

    dq_param_path = 'ResultsBO/dualquery_{}_{}_{}.csv'.format(args.dataset[0], args.workload[0], args.marginal[0])
    if not os.path.exists(dq_param_path):
        print('optimal parameters file for DQ does not exists!')
        dq_param_path = None
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
    support_path = 'private_support/{}_{}_{}'.format(args.dataset[0], args.workload[0], args.marginal[0])
    os.makedirs('private_support', exist_ok=True)
    os.makedirs(support_path, exist_ok=True)

    fem_data_fun = {}
    for run_id in range(total_runs):
        ith_support_path = f'{support_path}/r_{run_id}'
        os.makedirs(ith_support_path, exist_ok=True)
        np.random.seed(run_id)



        for e in epsilon_list:
            print(f'------> Running FEM and DQ: run {run_id}/{total_runs}, epsilon={e} in {epsilon_list}')
            # Dualquery
            dq_data_fun = dualquery.generate(real_ans, N, data.domain, query_manager, [e], delta,
                                             eta=3.36, samples=9, optimal_parameters_path=dq_param_path)

            # FEM
            fem_data_fun = fem.generate(real_ans, N, data.domain, query_manager, [e], delta,
                                        epsilon_split=0.01345, noise_multiple=0.099, samples=50, show_prgress=False)


            # Save FEM
            print(f'Saving data for epsilon={e}:')
            fem_data, fem_rho = fem_data_fun(e)
            fem_path = f'{ith_support_path}/fem_{e:.1f}.csv'
            fem_data.df.to_csv(fem_path, index=False)
            fem_rho_path = f'{ith_support_path}/fem_{e:.1f}.txt'
            rho_file = open(fem_rho_path, 'w')
            rho_file.write(str(fem_rho))
            rho_file.close()

            dq_data, dq_rho = dq_data_fun(e)
            dq_path = f'{ith_support_path}/dualquery_{e:.1f}.csv'
            dq_data.df.to_csv(dq_path, index=False)
            dq_rho_path = f'{ith_support_path}/dualquery_{e:.1f}.txt'
            rho_file = open(dq_rho_path, 'w')
            rho_file.write(str(dq_rho))
            rho_file.close()
            print(f'Saving {fem_path} and {dq_path}. fem_error={np.abs(real_ans - query_manager.get_answer(fem_data)).max():.3f}'
                  f'\tdq_error={np.abs(real_ans - query_manager.get_answer(dq_data)).max():.3f}')
