import itertools
import numpy as np

def kappa(data):
    if (len(data.shape) != 2):
        raise (ValueError, 'input must be 2-dimensional array')
    if (len(set(data.shape)) > 1):
        message = 'array dimensions must be N x N (they are {} x {})'
        raise (ValueError, message.format(*data.shape))
#    if not issubclass(data.dtype.type, np.integer):
#        raise (TypeError, 'expected integer type')
    if not np.isfinite(data).all():
        raise (ValueError, 'all data must be finite')
    if (data < 0).any():
        raise (ValueError, 'all data must be non-negative')
    if np.sum(data) <= 0:
        raise (ValueError, 'total data must sum to positive value')
    observation = observed(data)
    expectation = expected(data)
    perfection = 1.0
    k = np.divide(
        observation - expectation,
        perfection - expectation
    )
    return k

def observed(data):
    """Computes the observed agreement, Pr(a), between annotators."""
    total = float(np.sum(data))
    agreed = np.sum(data.diagonal())
    percent_agreement = agreed / total
    return percent_agreement

def expected(data):
    """Computes the expected agreement, Pr(e), between annotators."""
    total = float(np.sum(data))
    annotators = range(len(data.shape))
    percentages = ((data.sum(axis=i) / total) for i in annotators) 
    percent_expected = np.dot(*percentages)
    return percent_expected


def make_kappa_table(assign1, assign2):
    # 
    n1 = len(assign1)
    n2 = len(assign2)
    nb_c1 = np.amax(assign1) + 1
    nb_c2 = np.amax(assign2) + 1
    
    # Error checking
    if (n1 != n2):
        raise (ValueError, 'parameters size must be equal')

    table = np.zeros((max(nb_c1,nb_c1), max(nb_c1, nb_c2)))
    for i in range(n1):
        table[assign1[i]][assign2[i]] += 1
    return table

def kappa_perm(assign_0, assign_1):
    table = make_kappa_table(assign_0, assign_1)
    
    res = 0
    final_perm = table
    # print('iterations: ' + str(len(table)))
    for p in itertools.permutations(table):
        k = kappa(np.array(p))
        if (k > res):
            res = k
            final_perm = p
    return res, p
