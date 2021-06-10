import numpy as np
import random
import matplotlib.pyplot as plt

def monotonic_pairs(max_dist, seed = None):
    # This function will generate a list of pairs
    # List in form of (x,y), where y is monotonicly decreasing.
    # 0<=x<=1.   0<y.   List includes (1,0) and (0,1).
    # The function is assumed to be piecewise linear.
    # max_dist >= xi-x_(i-1)
    if seed is not None:
        random.seed(seed)
    return random_seq(max_dist, 0,1, 0,1)


def random_point(x_min, x_max, y_min, y_max):
    return (random.uniform(x_min, x_max), random.uniform(y_min, y_max))

def random_seq(max_dist, x_min, x_max, y_min, y_max):
    x_mid, y_mid= random_point(x_min, x_max, y_min, y_max)
    if x_max-x_min < max_dist:
        return [(x_mid, y_mid)]
    else:
        return random_seq(max_dist, x_min, x_mid, y_mid, y_max) + \
                    [(x_mid, y_mid)] + \
                    random_seq(max_dist, x_mid, x_max, y_min, y_mid)


def plot_pairs(pairs):
    x,y = pairs
    plt.plot(x,y)

def get_monotonic_vals(shells, smooth=0.0, seed=None):
    # This function will generate a random sequence with monotonic pairs,
    # then sample it to have exactly the number of points requested.
    # Linear interpolation.
    # @param num:  Number of points.
    #
    # Note that the last point will always be (1,0)

    # Make sure we have enough points.

    # TODO:   This is a hack
    # Can do better to make sure each point is separate.
    # Not sure it makes a difference.
    num = len(shells)
    pairs = monotonic_pairs(0.5/num, seed)

    x,y = zip(*pairs)
    
    def smooth_func(data, kernel_size):
        from scipy.signal import savgol_filter
        new_data =  savgol_filter(data, kernel_size, 3) # window size 51, polynomial order 3
        new_data[-1] = max(0, new_data[-1])
        for i in reversed(range(len(new_data)-1)):
            new_data[i] = max(new_data[i], new_data[i+1])
        return new_data

    if smooth > 0:
        y = smooth_func(y, smooth)
    return np.interp(shells,x,y)
