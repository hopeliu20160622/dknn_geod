import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def binary_search_helper(x, v, f, l, u, y=None, eps=1e-5, conf=None, cred=None):
    """
    Inputs
        x: data point
        v: mean of the cluster
        f: forward function, computes the label f(x)
        l: lower bound on delta
        u: upper bound on delta
    Returns
        s_delta: smallest delta which changes the label of x according to f
                 retuns -1 if no such label exists
        inf_norm: infinity norm of the attack vector
        conf: confidence levels of the perturbed image
        cred: credibility levels of the perturbed image
    """
    # To avoid any crazy manifolds where the label changes along multiple
    # decision boundaries, we shall keep a smallest delta
    s_delta = u # smallest delta
    inf_norm = np.linalg.norm((v - x)[:, :, 0], ord=np.inf) # assume the mean has different label
    attack_v = None
    y_w = None

    if y is None or conf is None or cred is None:
        y, conf, cred = f(x)
        y, conf, cred = y[0], np.max(conf), np.max(cred)

    steps = 0
    while u - l > eps:
        steps += 1
        delta = l + (u - l)/2
        attack_vector = np.multiply(delta, v - x)
        xp = x + attack_vector # move x towards v
        yp, confp, credp = f(xp) # predict labels using f
        yp, confp, credp = yp[0], np.max(confp), np.max(credp)

        if y != yp: # attack worked
            u = delta
            if s_delta >= delta: # update
                s_delta = delta
                conf = confp
                cred = credp
                inf_norm = np.linalg.norm(attack_vector[:, :, 0], ord=np.inf)
                attack_v = attack_vector
                y_w = yp
        else:
            l = delta

    #print('Binary seach took', steps, 'steps')
    #print('Found delta', s_delta)
    #print('Inf norm', inf_norm)
    #print('Conf', conf)
    #print('Cred', cred)
    return s_delta, inf_norm, conf, cred

def binary_search(x, v, f, y=None, eps=1e-5, conf=None, cred=None):
    """ See binary_search_helper """
    return binary_search_helper(x, v, f, 0, 1, y, eps, conf=conf, cred=cred)

def get_deltas(X, Y, means, labels, f, eps=1e-5):
    """
    Inputs
        X: data points
        Y: true labels for X
        means: means of the cluster corresponding to some label
        labels: same length as means; means[i] is the mean of points w labels[i]
        f: function which takes a single point x and predicts y
    """
    assert(len(labels) == len(means))
    assert(len(X) == len(Y))

    delta_list = []
    inf_norm_list = []
    conf_list = []
    cred_list = []

    for x, y in tqdm(zip(X, Y), total=X.shape[0]):
        # to avoid computing smallest delta for every mean, we use the
        # heuristic of finding the nearest mean from x (which has different
        # label) and compute the smallest delta for that mean only
        ypred, conf, cred = f(x)  # compare against the model prediction
        ypred, conf, cred = ypred[0], np.max(conf), np.max(cred)

        if y != ypred:
            #print('wrong prediction')
            delta_list.append(0)
            inf_norm_list.append(0)
            conf_list.append(conf)
            cred_list.append(cred)
            continue

        temp_means = means.copy()
        self_label_index = np.argwhere(labels==y)
        temp_means[self_label_index] = np.full((means[self_label_index].shape), np.inf)
        nearest_mean = np.argmin(np.linalg.norm(x.flatten()-temp_means.reshape((temp_means.shape[0], -1)), axis=1))
        delta, inf_norm, conf, cred = binary_search(x, means[nearest_mean], f, y, eps, conf=conf, cred=cred)

        delta_list.append(delta)
        inf_norm_list.append(inf_norm)
        conf_list.append(conf)
        cred_list.append(cred)

    return np.asarray(delta_list), np.asarray(inf_norm_list), np.asarray(conf_list), np.asarray(cred_list)
