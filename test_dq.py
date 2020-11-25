from algorithms import dualquery
from Util.qm import QueryManager
from Util import benchmarks


dataset = 'adult'
workload = 64
marginal =  3
epsilon = [0.5]

# Get dataset
data, workloads = benchmarks.randomKway(dataset, workload, marginal)
N = data.df.shape[0]
delta = 1.0 / N ** 2
dq_param_path = 'ResultsBO/dualquery_adult_64_3.csv'

# Get Queries
query_manager = QueryManager(data.domain, workloads)
print("Number of queries = ", len(query_manager.queries))
print('computing real answers...')
real_answers = query_manager.get_answer(data)
dq_data_fun = dualquery.generate(real_answers, N, data.domain, query_manager, [0.5], delta,
                                             eta=3.36, samples=9, optimal_parameters_path=dq_param_path)
