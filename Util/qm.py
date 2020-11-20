import numpy as np
import itertools
from collections.abc import Iterable
from datasets.dataset import Dataset
from datasets.domain import Domain

class QueryManager():
    """< 1e-9
    K-marginal queries manager
    """
    def __init__(self, domain, workloads):
        self.real_answers = None
        self.domain = domain
        self.workloads = workloads
        self.att_id = {}
        col_map = {}
        for i,col in enumerate(self.domain.attrs):
            col_map[col] = i
            self.att_id[col] = i
        feat_pos = []
        cur = 0
        for f, sz in enumerate(domain.shape):
            feat_pos.append(list(range(cur, cur + sz)))
            cur += sz
        self.dim = np.sum(self.domain.shape)
        self.queries = []
        for feat in self.workloads:
            f_sz = np.zeros(len(feat))
            positions = []
            for col in feat:
                i = col_map[col]
                positions.append(feat_pos[i])
            for tup in itertools.product(*positions):
                self.queries.append(tup)
        self.num_queries = len(self.queries)

    def get_small_separator_workload(self):
        W = []
        for i in range(self.dim):
            w = np.zeros(self.dim)
            w[i] = 1
            W.append(w)
        return np.array(W)

    def get_query_workload(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        W = []
        for q_id in q_ids:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                w[p] = 1
            W.append(w)
        if len(W) == 1:
            W = np.array(W).reshape(1, -1)
        else:
            W = np.array(W)
        return W

    def get_query_workload_weighted(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        wei = {}
        for q_id in q_ids:
            wei[q_id] = 1 + wei[q_id] if q_id in wei else 1
        W = []
        weights = []
        for q_id in wei:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                w[p] = 1
            W.append(w)
            weights.append(wei[q_id])
        if len(W) == 1:
            W = np.array(W).reshape(1,-1)
        else:
            W = np.array(W)
        return W, weights

    def get_answer(self, data, weights=None, normalize=True):
        ans_vec = np.array([])
        N_sync = data.df.shape[0]
        # for proj, W in self.workloads:
        for proj in self.workloads:
            # weights let's you do a weighted sum
            x = data.project(proj).datavector(weights=weights)
            if weights is None and normalize:
                x = x / N_sync
            ans_vec = np.append(ans_vec, x)
        return ans_vec

    def get_query_matrix(self, support_dataset: Dataset, domain: Domain, q_ids: list):
        Q = []
        for row in support_dataset.df.iterrows():
            row_arr = row[1].values
            row_dataset = Dataset([row_arr], domain)
            answers = self.get_answer(row_dataset)
            Q.append(answers)
        Q = np.array(Q).T
        return Q

