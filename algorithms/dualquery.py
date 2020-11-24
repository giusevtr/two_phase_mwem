import numpy as np
import sys,os
import pandas as pd
from datasets.dataset import Dataset
from datasets.dataset import Domain
from Util import oracle_dq, util2, benchmarks

from GPyOpt.methods import BayesianOptimization
from tqdm import tqdm
from Util.qm import QueryManager

def search_T(eps, n, delta, eta, s):
    lo = 0
    hi = 100000
    for _ in range(200):
        temp_T = (lo + hi) // 2
        temp_eps = (2*eta*(temp_T-1)/n)*(np.sqrt(2*s*(temp_T-1)*np.log(1/delta)) + s*(temp_T-1)*(np.exp(2*eta*(temp_T-1)/n)-1))
        # print("T={}   t_eps={:.5f}".format(temp_T, temp_eps))
        if temp_eps <= eps:
            lo = temp_T
        else:
            hi = temp_T
    return lo


def generate(real_answers:np.array,
             N:int,
             domain: Domain,
             query_manager: QueryManager,
             epsilon: list,
             delta: float,
             eta=1.2, samples=20, optimal_parameters_path=None):
    D = np.sum(domain.shape)
    Q_size = query_manager.get_num_queries()
    Q_dist = np.ones(2*Q_size)/(2*Q_size)

    """
    Read parameters
    """
    alpha = 0

    X = []

    neg_real_answers = 1 - real_answers

    cumulative_rho = 0
    epsilon_index = 0
    T = 0
    cumulative_rho_at_time_t = {}

    eta_t = {}
    samples_t = {}
    while True:
        if optimal_parameters_path is not None:
            # Find the best parameters for the current epsilon
            param_df = pd.read_csv(optimal_parameters_path)
            epsilon_params = param_df[param_df['epsilon'] == epsilon[epsilon_index]]
            if len(epsilon_params) == 0: continue
            eta = epsilon_params.min()['eta']
            samples = int(epsilon_params.min()['samples'])
            print(f'updating: at time={T} eta={eta} and samples={samples}')
        eta_t[T] = eta
        samples_t[T] = samples

        # Multiplicative weights
        epsilon_0 = 2 * eta_t[T] * T / N
        rho_0 = util2.from_eps_to_rho(epsilon_0)
        cumulative_rho += samples_t[T]*rho_0
        cumulative_rho_at_time_t[T] = cumulative_rho
        T = T + 1

        current_epsilon = util2.from_rho_to_eps_delta(cumulative_rho, delta)
        if epsilon[epsilon_index] < current_epsilon:
            epsilon_index += 1
            if epsilon_index == len(epsilon): break

    for t in range(T):
        """
        get s samples
        """
        queries = []
        neg_queries = []
        for _ in range(samples_t[t]):
            q_id = util2.sample(Q_dist)
            if q_id < Q_size:
                queries.append(q_id)
            else:
                neg_queries.append(q_id-Q_size)
        # query_ind_sample = [sample(Q_dist) for _ in range(s)]


        """
        Gurobi optimization: argmax_(x^t) A(x^t, q~)  >= max_x A(x, q~) - \alpha
        """
        # x, mip_gap = query_manager.solve(alpha, query_ind_sample, dataset.name)
        query_workload = query_manager.get_query_workload(queries)
        neg_query_workload = query_manager.get_query_workload(neg_queries)
        oh_fake_data = oracle_dq.dualquery_best_response(query_workload, neg_query_workload, D, domain, alpha)
        # max_mip_gap = max(max_mip_gap, mip_gap)
        X.append((cumulative_rho_at_time_t[t], oh_fake_data))

        """
        ## Update query player distribution using multiplicative weights
        """
        fake_data = Dataset(pd.DataFrame(util2.decode_dataset(oh_fake_data, domain), columns=domain.attrs), domain)
        fake_answers = query_manager.get_answer(fake_data)
        neg_fake_answers = 1 - fake_answers
        A = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)
        Q_dist = np.exp(eta_t[t]*A)*Q_dist

        """
        ## Normalize
        """
        sum = np.sum(Q_dist)
        Q_dist = Q_dist / sum

        assert np.abs(np.sum(Q_dist)-1)<1e-6, "Q_dist must add up to 1"

        # util2.progress_bar(epsilon, curr_eps, msg="dualquery: t={}".format(t))

    # fake_data = Dataset(pd.DataFrame(util2.decode_dataset(X, domain), columns=domain.attrs), domain)
    print("")
    # print("max_mip_gap = ", max_mip_gap)
    # return {"X":fake_data}
    def data_fun(final_epsilon):
        data = []
        final_rho = 0
        for rho, syndata_row in X:
            this_epsilon = util2.from_rho_to_eps_delta(rho, delta)
            final_rho = rho
            if this_epsilon > final_epsilon: break
            data.append(syndata_row)
        dataset = Dataset(pd.DataFrame(util2.decode_dataset(data, domain), columns=domain.attrs), domain)
        return dataset, final_rho
    return data_fun


def bo_search(real_answers:np.array,
                  N: int,
                  domain: Domain,
                  query_manager: QueryManager,
                  epsilon: float,
                  delta: float,
                  eta_range: tuple,   # should be in (0.5, 5)
                  samples_range: tuple,  # should be in (5, 200)
                  bo_iters=25):
    bo_domain = [{'name': 'eta', 'type': 'continuous', 'domain': eta_range},
                 {'name': 'samples', 'type': 'continuous', 'domain': samples_range}]

    progress = tqdm(total=bo_iters, desc='BO-DQ')

    def get_error(params):
        eta = params[0][0]
        samples = int(params[0][1])
        # print(eta, samples)
        data_func = generate(real_answers=real_answers,
                                     N=N,
                                     domain=domain,
                                     query_manager=query_manager,
                                     epsilon=[epsilon],
                                     delta=delta,
                                     eta=eta,
                                     samples=samples,
                                     )
        syndata, _ = data_func(epsilon)
        max_error = np.abs(real_answers - query_manager.get_answer(syndata)).max()
        progress.update()
        progress.set_postfix({f'error({eta:.4f}, {samples:.2f})': max_error})
        return max_error

    # --- Solve your problem
    myBopt = BayesianOptimization(f=get_error, domain=bo_domain, exact_feval=False)
    myBopt.run_optimization(max_iter=bo_iters)
    # myBopt.plot_acquisition()
    eta= myBopt.x_opt[0]
    samples = myBopt.x_opt[1]
    min_error = myBopt.fx_opt

    names = ["epsilon", "bo_iters", "eta", "samples", "error"]
    res = [[epsilon, bo_iters, eta, samples, min_error]]
    return pd.DataFrame(res, columns=names)



if __name__ == "__main__":
    # description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    # parser.add_argument('--eps_split_lo', type=float, default=0.01, help='eps0 parameter range')
    # parser.add_argument('--eps_split_hi', type=float, default=0.05, help='eps0 parameter range')
    # parser.add_argument('--noise_mult_lo', type=float, default=0.01, help='noise parameter range')
    # parser.add_argument('--noise_mult_hi', type=float, default=0.1, help='noise parameter range')
    args = parser.parse_args()
    # print(vars(args))
    dataset = args.dataset[0]
    workload = args.workload[0]
    marginal = args.marginal[0]
    epsilon = [args.epsilon[0]]
    eta_low = 0.5
    eta_hi = 7
    samples_low = 2
    samples_hi = 50
    bo_iters = 100

    # Get dataset
    data, workloads = benchmarks.randomKway(dataset, workload, marginal)
    N = data.df.shape[0]
    delta = 1.0/N**2

    # Get Queries
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    print('computing real answers...')
    real_answers = query_manager.get_answer(data)
    print('Done!')
    final_df = None
    for eps in epsilon:
        print("epsilon = ", eps, "=========>")
        # Generate synthetic data with eps
        # df = fem_bo_search(data, query_manager, eps, tuple(args.eps0), tuple(args.noise))
        df = bo_search(real_answers=real_answers,
                           N=N,
                           domain=data.domain,
                           query_manager=query_manager,
                           epsilon=eps,
                           delta=delta,
                           eta_range=(eta_low, eta_hi),
                           samples_range=(samples_low, samples_hi),
                           bo_iters=bo_iters)

        if final_df is None:
            final_df = df
        else:
            final_df = final_df.append(df)
    file_name = "ResultsBO/dualquery_{}_{}_{}.csv".format(dataset, workload, marginal)
    print("Saving ", file_name)
    if os.path.exists(file_name):
        dfprev = pd.read_csv(file_name)
        final_df = final_df.append(dfprev, sort=False)
    os.makedirs('ResultsBO', exist_ok=True)
    final_df.to_csv(file_name, index=False)
