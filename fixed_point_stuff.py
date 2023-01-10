import numpy as np
import numba
from numba import jit

@numba.njit
def square(x):
    return np.square(x)

@numba.njit
def skew(x):
    return x**3/3

@numba.njit
def exp(x):
    return np.exp(-np.square(x)/2)

@numba.njit
def logcosh(x):
    return np.log(np.cosh(x))

@numba.njit
def dot_square(x):
    return 2*x

@numba.njit
def dot_skew(x):
    return 2*(np.square(x))/3

@numba.njit
def dot_exp(x):
    return -1*(np.exp(-np.square(x)/2)) + np.dot((np.square(x)), np.exp(-np.square(x)/2))

@numba.njit
def dot_logcosh(x):
    return np.tanh(x)

@numba.njit(fastmath=True)
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@numba.njit
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)


@numba.njit(fastmath=True)
def fixed_point_alg(w_n, B, Z,cf, dot_cf, its = 500):

    """ Update function for source separation vectors. The code user can select their preferred contrast function using a string input:
    1) square --> x^2
    2) logcosh --> log(cosh(x))
    3) exp  --> e(-x^2/2)
    4) skew --> -x^3/3 
    e.g. skew is faster, but less robust to outliers relative to logcosh and exp
    Upon meeting a threshold difference between iterations of the algorithm, separation vectors are discovered 
    
    The maximum number of iterations (its) and the contrast function type (cf) are already specified, unless alternative input is provided. """
   
    assert B.ndim == 2
    assert Z.ndim == 2
    assert w_n.ndim == 1
    assert its in [500]

    counter = 0
    its_tolerance = 0.0001
    sep_diff = np.ones(its)
    B_T_B = B @ B.T

    while sep_diff[counter] > its_tolerance or counter >= its:

        # transfer current separation vector as the previous arising separation vector
        w_n_1 = w_n.copy()
        # use update function to get new, current separation vector
        A = dot_cf(np.dot(w_n_1.T, Z)).mean()
        #w_n = np.expand_dims((Z*cf(np.dot(np.transpose(w_n_1), Z))).mean(axis=1), axis=1) - A * w_n_1
        w_n = np_mean(Z * cf(w_n_1 @ Z), axis = 1) - A * w_n_1
        # orthogonalise separation vectors
        w_n -= np.dot(B_T_B, w_n)
        # normalise separation vectors
        w_n /= np.linalg.norm(w_n)
        counter += 1
        sep_diff[counter] = np.abs(w_n @ w_n_1 - 1)

    return w_n



def get_spikes(w_n,Z, fsamp):

    """ Based on gradient convolutive kernel compensation. Aim to remove spurious discharges to improve the source separation
    vector estimate. Results in a reduction in ISI vairability (by seeking to minimise the covariation in MU discharges)"""

    # Step 4a: 
    #source_pred = np.square(np.dot(np.transpose(w_n),Z)).real # element-wise square of the input to estimate the ith source
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = fsamp*0.02 ) # peaks variable holds the indices of all peaks
    #source_pred /= np.mean(source_pred[:,np.argpartition(source_pred[:,peaks], -10)[-10:]]) 
    #print(np.shape(source_pred))
    print(np.shape(source_pred))
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        #print(np.shape(spikes))
        # remove outliers from the spikes cluster with a std-based threshold
        #spikes = spikes[source_pred[:,spikes] < np.mean(source_pred[:,spikes],axis=1) +3*np.std(source_pred[spikes],axis=1)]
    else:
        spikes = peaks

    return source_pred, spikes