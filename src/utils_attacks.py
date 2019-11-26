import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def binary_search_helper(x, v, f, l, u, y=None, eps=1e-5):
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
    """
    # To avoid any crazy manifolds where the label changes along multiple
    # decision boundaries, we shall keep a smallest delta
    s_delta = u # smallest delta
    inf_norm = np.linalg.norm((v - x)[:, :, 0], ord=np.inf) # assume the mean has different label
    attack_v = None
    y_w = None

    if y is None:
        y = f(x)[0]

    steps = 0
    temp = f(v)[0][0]
    #print('true y is', y)
    #print('other mean y is', temp)
    while u - l > eps:
        steps += 1
        delta = l + (u - l)/2
        attack_vector = np.multiply(delta, v - x)
        xp = x + attack_vector # move x towards v
        yp = f(xp)[0][0] # predict labels using f
        #print('yp is', yp)

        if y != yp: # attack worked
            u = delta
            if s_delta >= delta: # update smallest delta and attack norm
                s_delta = delta
                inf_norm = np.linalg.norm(attack_vector[:, :, 0], ord=np.inf)
                attack_v = attack_vector
                y_w = yp
        else:
            l = delta
    
    #fig, axes = plt.subplots(1, 3)
    #ax0, ax1, ax2 = axes.flatten()
    #ax0.imshow(x[:, :, 0], cmap='gray')
    #ax0.set_title('original; label='+str(y))
    #if y_w is not None:
    #    ax1.imshow((x+attack_v)[:, :, 0], cmap='gray')
    #    ax1.set_title('perturbed; label='+str(y_w))
    #else:
    #    print('No attack worked')
    #ax2.imshow(v[:, :, 0], cmap='gray')
    #ax2.set_title('closest other mean='+str(temp))
    #plt.show()

    #print('Binary seach took', steps, 'steps')
    #print('Found delta', s_delta)
    #print('Inf norm', inf_norm)
    return s_delta, inf_norm

def binary_search(x, v, f, y=None, eps=1e-5):
    """ See binary_search_helper """
    return binary_search_helper(x, v, f, 0, 1, y, eps)

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
    for x, y in tqdm(zip(X, Y), total=X.shape[0]):
        # to avoid computing smallest delta for every mean, we use the
        # heuristic of finding the nearest mean from x (which has different
        # label) and compute the smallest delta for that mean only
        ypred = f(x)[0]  # compare against the model prediction
        if y != ypred:
            #print('wrong prediction')
            delta_list.append(0)
            inf_norm_list.append(0)
            continue
        temp_means = means.copy()
        self_label_index = np.argwhere(labels==y)
        temp_means[self_label_index] = np.full((means[self_label_index].shape), np.inf)
        nearest_mean = np.argmin(np.linalg.norm(x.flatten()-temp_means.reshape((temp_means.shape[0], -1)), axis=1))
        delta, inf_norm = binary_search(x, means[nearest_mean], f, y, eps)
        delta_list.append(delta)
        inf_norm_list.append(inf_norm)
    return np.asarray(delta_list), np.asarray(inf_norm_list)
