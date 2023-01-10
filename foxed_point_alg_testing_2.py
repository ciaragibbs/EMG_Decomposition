import numpy as np

def logcosh(x):
    return np.log(np.cosh(x))


def dot_logcosh(x):
    return np.tanh(x)



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


def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

cf = logcosh
dot_cf = dot_logcosh

w_n = np.random.rand(1000)
B = np.random.rand(1000, 300)
Z = np.random.rand(1000, 40000)
Z_meaner = Z.shape[1]


w_n_1 = w_n.copy()
wTZ = w_n_1.T @ Z 

w_n_a = np_mean(Z * cf(wTZ), axis = 1) 

w_n_b = Z @ cf(wTZ).T /Z_meaner  
        
print(np.allclose(w_n_a,w_n_b))
