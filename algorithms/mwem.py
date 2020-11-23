import numpy as np
from tqdm import tqdm
from algorithms import exponential_mechanism
from datasets.dataset import Dataset
from Util.qm import QueryManager
from Util import util2
from datasets.dataset import Dataset
from datasets.domain import Domain
from Util import benchmarks
import argparse

def get_data_onehot(data):
    df_data = data.df.copy()
    dim = np.sum(data.domain.shape)
    i = 0
    for attr in data.domain.attrs:
        df_data[attr] += i
        i += data.domain[attr]
    data_values = df_data.values
    data_onehot = np.zeros((len(data_values), dim))
    arange = np.arange(len(data_values))
    arange = np.tile(arange, (data_values.shape[1], 1)).T
    data_onehot[arange, data_values] = 1
    return data_onehot


def get_A_init(start_data: Dataset):
    df_pub = start_data.df
    cols = list(df_pub.columns)
    df_pub = df_pub.groupby(cols).size().reset_index(name='Count')
    A_init = df_pub['Count'].values
    A_init = A_init / A_init.sum()
    data_pub = Dataset(df_pub, start_data.domain)
    return data_pub, A_init


'''
satisfies rho-zCDP
'''
def generate(start_data: Dataset,
             real_answers:np.array,
             N: int,
             query_manager: QueryManager,
             epsilon: float,
             delta: float,
             epsilon_split: float,
             cumulative_rho: float = 0,
             return_last=False):

    data_support, A_init = get_A_init(start_data)
    data_onehot = get_data_onehot(data_support)

    # initialize A to be uniform distribution over support of data/fake_data
    A = np.copy(A_init)

    # initialize A_avg so that we can take an average of all A_t's at the end
    A_avg = np.copy(A_init)

    epsilon_0 = epsilon * epsilon_split
    rho_0 = util2.from_eps_to_rho(epsilon_0)
    # Note: eps0/2 since we compose 2 mechanisms and need it to add up to eps0
    # for t in tqdm(range(T), desc='WMEM'):
    T = 0
    while util2.from_rho_to_eps_delta(cumulative_rho + rho_0, delta) < epsilon:
        T += 1
        cumulative_rho += rho_0

        fake_answers = query_manager.get_answer(data_support, weights=A)

        # 1) Exponential Mechanism
        score = np.abs(real_answers - fake_answers)
        # print(f'error({T}) = {np.max(score)}, rho={cumulative_rho},   eps0={epsilon_0},   epsilon={epsilon}')
        # eps0 = np.sqrt(2*rho_0)
        q_t_ind = exponential_mechanism.sample(score, N, eps0=epsilon_0/2)

        # 2) Laplacian Mechanism
        m_t = real_answers[q_t_ind]
        m_t += np.random.laplace(loc=0, scale=(2 / (N * epsilon_0))) # Note: epsilon_0 = eps / T, sensitivity is 1/N
        m_t = np.clip(m_t, 0, 1)

        # 3) Multiplicative Weights update
        query = query_manager.get_query_workload([q_t_ind])
        q_t_x = data_onehot.dot(query.T).flatten()
        q_t_x = (q_t_x == query.sum()).astype(int)
        q_t_A = fake_answers[q_t_ind]

        factor = np.exp(q_t_x * (m_t - q_t_A)) # check if n times distribution matters - / (2 * N)
        A = A * factor
        A = A / A.sum()
        A_avg += A

    A_avg /= (T+1)
    assert np.abs(np.sum(A_avg) - 1) < 0.0001, 'A_avg is not a distribution'
    if return_last:
        return A_avg, A
    return data_support, A_avg



if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('--epsilon_split', type=float, default=0.02, help='eps0 hyperparameter')
    args = parser.parse_args()

    print("=============================================")
    print(vars(args))

    ######################################################
    ## Get dataset
    ######################################################
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]
    delta = 1.0/N**2

    ######################################################
    ## Get Queries
    ######################################################
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))

    real_ans = query_manager.get_answer(data)

    V = real_ans > 0.05
    support_size = np.sum(V)
    ######################################################
    ## Generate synthetic data with eps
    ######################################################
    rand_data = Dataset.synthetic(data.domain, N=support_size)
    samp_data = data.sample(s=support_size)
    rand_data.df.drop_duplicates(inplace=True)

    samp_data.df.drop_duplicates(inplace=True)
    samp_support_error = np.abs(real_ans - query_manager.get_answer(samp_data)).max()
    print(f'samp_data size = {len(samp_data.df)} error = {samp_support_error}')

    print("epsilon\trandom_supp error\tsampled_supp error")
    for eps in args.epsilon:
        rand_supp, rand_dist = generate(start_data=rand_data, real_answers=real_ans, N=N, query_manager=query_manager, epsilon=eps, delta=delta, epsilon_split=args.epsilon_split)
        samp_supp, samp_dist = generate(start_data=samp_data, real_answers=real_ans, N=N, query_manager=query_manager, epsilon=eps, delta=delta, epsilon_split=args.epsilon_split)

        max_error_random_support = np.abs(real_ans - query_manager.get_answer(rand_supp, rand_dist)).max()
        max_error_sampled_support = np.abs(real_ans - query_manager.get_answer(samp_supp, samp_dist)).max()
        print("{}\t{:.5f}\t{:.5f},".format(eps, max_error_random_support, max_error_sampled_support))
