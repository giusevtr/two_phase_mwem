from Util import util2
import numpy as np


def sample(score, N, eps0):
    EM_dist_0 = np.exp(eps0 * score * N / 2, dtype=np.float128)  # Note: sensitivity is 1/N
    EM_dist = EM_dist_0 / EM_dist_0.sum()
    q_t_ind = util2.sample(EM_dist)
    return q_t_ind