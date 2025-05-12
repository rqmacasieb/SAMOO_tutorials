import numpy as np
from scipy.stats import norm

def first_dom_second(first, second, beta, ppd=True):
    mean1_obj1, s2_1_obj1, mean1_obj2, s2_1_obj2 = first
    mean2_obj1, s2_2_obj1, mean2_obj2, s2_2_obj2 = second
    if ppd:
        pd_obj1 = norm.cdf((mean2_obj1 - mean1_obj1) / np.sqrt(s2_1_obj1 + s2_2_obj1))
        pd_obj2 = norm.cdf((mean2_obj2 - mean1_obj2) / np.sqrt(s2_1_obj2 + s2_2_obj2))
        return pd_obj1 * pd_obj2 >= beta
    else:
        return mean1_obj1 < mean2_obj1 and mean1_obj2 < mean2_obj2


def dominance_eval(obj1_pred, obj2_pred, beta, n_samples=None, ppd = True):
    if n_samples is None:
        n_samples = len(obj1_pred)
        
    dominated_count = np.zeros(n_samples)
    means_obj1 = np.array([obj1_pred[i]['mean'] for i in range(n_samples)])
    s2_obj1 = np.array([obj1_pred[i]['s2'] for i in range(n_samples)])
    means_obj2 = np.array([obj2_pred[i]['mean'] for i in range(n_samples)])
    s2_obj2 = np.array([obj2_pred[i]['s2'] for i in range(n_samples)])
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                continue
            if first_dom_second(
                [means_obj1[j], s2_obj1[j], means_obj2[j], s2_obj2[j]],
                [means_obj1[i], s2_obj1[i], means_obj2[i], s2_obj2[i]],
                beta, ppd
            ):
                dominated_count[i] += 1
                
    return dominated_count

def get_non_dominated_points(obj1_pred, obj2_pred, beta, n_samples=None, ppd = True):
    domcount = dominance_eval(obj1_pred, obj2_pred, beta, n_samples, ppd)
    min_domcount = np.min(domcount)
    return np.where(domcount == min_domcount)[0]