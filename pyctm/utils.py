import numpy as np
import scipy
from pyctm.inferencer import compute_dirichlet_expectation


def topic_beta(mdl):
    """Get a list of dicts with topics and words related (pandas-dataframable)."""
    E_log_eta = compute_dirichlet_expectation(mdl._eta)
    ll = []
    for topic_index in range(mdl._number_of_topics):
        beta_probability = np.exp(
            E_log_eta[topic_index, :] -
            scipy.special.logsumexp(E_log_eta[topic_index, :]))

        dd = {}
        for type_index in reversed(np.argsort(beta_probability)):
            dd[mdl._index_to_type[type_index]] = beta_probability[type_index]
        ll.append(dd)
    return ll
