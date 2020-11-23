import numpy as np
import pandas as pd
import itertools
from collections.abc import Iterable
from datasets.dataset import Dataset
from datasets.domain import Domain
from tqdm import tqdm


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
        feat_size = []
        cur = 0
        for f, sz in enumerate(domain.shape):
            feat_pos.append(list(range(cur, cur + sz)))
            feat_size.append(list(range(0, sz)))
            cur += sz
        self.dim = np.sum(self.domain.shape)
        self.queries = []
        self.queries_2 = []
        for feat in self.workloads:
            f_sz = np.zeros(len(feat))
            positions = []
            values = []
            for col in feat:
                i = col_map[col]
                positions.append(feat_pos[i])
                values.append(feat_size[i])
            for tup in itertools.product(*positions):
                self.queries.append(tup)
            for tup in itertools.product(*values):
                self.queries_2.append(tup)

        num_queries = len(self.queries)
        self.active_query = np.ones(num_queries)
        self.active_query_ids = np.arange(num_queries)

    def get_num_queries(self):
        return np.sum(self.active_query, dtype=np.int)

    def set_queries(self, ids):
        self.active_query[:] = 0
        self.active_query[ids] = 1
        self.active_query_ids = ids
        # self.active_query_ids.astype(int)

    def get_queries_2(self):
        q2 = []
        for i in self.active_query_ids:
            q2.append(self.queries_2[i])
        return q2

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
        assert np.max(q_ids) < self.get_num_queries()
        for i in q_ids:
            q_id = self.active_query_ids[i]
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
        assert np.max(q_ids) < self.get_num_queries()
        wei = {}
        for i in q_ids:
            q_id = self.active_query_ids[i]
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
        return ans_vec[self.active_query_ids]

    def get_answer_for_few_queries(self, data, weights=None, normalize=True):
        ans_vec = np.array([])
        N_sync = data.df.shape[0]
        # for proj, W in self.workloads:
        for proj in self.workloads:
            # weights let's you do a weighted sum
            x = data.project(proj).datavector(weights=weights)
            if weights is None and normalize:
                x = x / N_sync
            ans_vec = np.append(ans_vec, x)
        return ans_vec[self.active_query_ids]

    def get_query_matrix(self, support_dataset: Dataset, domain: Domain):
        Q = []
        for row in tqdm(support_dataset.df.iterrows(), desc='generation query matrix'):
            row_df = pd.DataFrame([row[1].values], columns=domain.attrs)
            row_dataset = Dataset(row_df, domain)
            answers = self.get_answer(row_dataset)
            # answers2 = answers[q_ids]
            Q.append(np.append([1], answers))
        Q = np.array(Q).T
        return Q

