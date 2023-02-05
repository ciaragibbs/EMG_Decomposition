import numpy as np
from scipy import signal
import scipy
import numpy as np
from sklearn.cluster import KMeans
import sklearn 
from sklearn.metrics import silhouette_samples
np.random.seed(1337)
signal = np.random.random((3000,1000)).T

def fixed_point_alg(w_n, B, Z,its = 500):

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
    Z_meaner = Z.shape[1]

    while sep_diff[counter] > its_tolerance or counter >= its:

        # transfer current separation vector as the previous arising separation vector
        w_n_1 = w_n.copy()
        # use update function to get new, current separation vector
        wTZ = w_n_1.T @ Z 
        A = 2*(wTZ).mean()
        w_n = Z @ ((wTZ)**2).T / Z_meaner  - A * w_n_1
        #w_n = np_mean(Z * cf(wTZ), axis = 1) - A * w_n_1
        #print(np.allclose(w_n,other_w_n))

        # orthogonalise separation vectors
        w_n -= np.dot(B_T_B, w_n)
        # normalise separation vectors
        w_n /= np.linalg.norm(w_n)
        counter += 1
        sep_diff[counter] = np.abs(w_n @ w_n_1 - 1)
    print(counter)
    return w_n



def get_spikes(w_n,Z, fsamp=2048):

    """ Based on gradient convolutive kernel compensation. Aim to remove spurious discharges to improve the source separation
    vector estimate. Results in a reduction in ISI vairability (by seeking to minimise the covariation in MU discharges)"""

    # Step 4a: 
    source_pred = np.square(np.dot(np.transpose(w_n),Z)).real # element-wise square of the input to estimate the ith source
    #source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    #source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1 ) # peaks variable holds the indices of all peaks
    print(peaks)
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


def min_cov_isi(w_n,B,Z,fsamp,cov_n,spikes_n): 
    
    cov_n_1 = 2 * cov_n
    while cov_n < cov_n_1:

        cov_n_1 = cov_n.copy()
        spikes_n_1 = spikes_n.copy()
        # w_n = np.expand_dims(w_n,axis=1)
        w_n_1 = w_n.copy()
        _ , spikes = get_spikes(w_n,Z,fsamp)
        # determine the interspike interval
        ISI = np.diff(spikes/fsamp)
        # determine the coefficient of variation
        CoV = np.std(ISI)/np.mean(ISI)
        # update the sepearation vector by summing all the spikes
        w_n = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
       

    # if you meet the CoV minimisation condition, but with a single-spike-long train, use the updated
    # separation vector and recompute the spikes

    if len(spikes) < 2:
        _ , spikes = get_spikes(w_n,Z,fsamp)

    return w_n_1, spikes_n_1



def get_silohuette(w_n,Z,fsamp):

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred /= max(source_pred)
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1 ) # this is approx a value of 20, which is in time approx 10ms
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        #sil1 = sklearn.metrics.silhouette_score(source_pred[peaks].reshape(-1,1), kmeans.labels_, metric='euclidean')
        #sil = (sil1[kmeans.labels_ == 1].mean() + sil1[kmeans.labels_ == 2].mean()) / 2
        silhouette_values = silhouette_samples(source_pred[peaks].reshape(-1,1), kmeans.labels_, metric='euclidean')
        mean_silhouette_score_cluster1 = silhouette_values[kmeans.labels_ == 0].mean()
        print(mean_silhouette_score_cluster1)
        mean_silhouette_score_cluster2 = silhouette_values[kmeans.labels_ == 1].mean()
        print(mean_silhouette_score_cluster2)
        sil = (mean_silhouette_score_cluster1 + mean_silhouette_score_cluster2) / 2
    
    
    else:

        sil = 0

    return source_pred, spikes, sil

def gausswin(M, alpha=2.5):
    """ Python equivalent of the in-built gausswin function MATLAB (since there is no open-source Python equivalent) """
    n = np.arange(-(M-1) / 2, (M-1) / 2 + 1,dtype=np.float128)
    w = np.exp((-1/2) * (alpha * n / ((M-1) / 2)) ** 2)
    return w

def cutMUAP(MUPulses, length, Y):

    """ Direct converion of MATLAB code in-lab. Extracts consecutive MUAPs out of signal Y and stores
    them row-wise in the out put variable MUAPs.
    Inputs: 
    - MUPulses: Trigger positions (in samples) for rectangualr window used in extraction of MUAPs
    - length: radius of rectangular window (window length = 2*len +1)
    - Y: Single signal channel (raw vector containing a single channel of a recorded signals)
    Outputs:
    - MUAPs: row-wise matrix of extracted MUAPs (algined signal intervals of length 2*len+1)"""
 
    while len(MUPulses) > 0 and MUPulses[-1] + 2 * length > len(Y):
        MUPulses = MUPulses[:-1]

    c = len(MUPulses)
    edge_len = round(length / 2)
    tmp = gausswin(2 * edge_len) # gives the same output as the in-built gausswin function in MATLAB
    # create the filtering window 
    win = np.ones(2 * length + 1)
    win[:edge_len] = tmp[:edge_len]
    win[-edge_len:] = tmp[edge_len:]
    MUAPs = np.empty((c, 1 + 2 * length))
    for k in range(c):
        start = max(MUPulses[k] - length, 1) - (MUPulses[k] - length)
        end = MUPulses[k] + length- min(MUPulses[k] + length, len(Y))
        MUAPs[k, :] = win * np.concatenate((np.zeros(start), Y[max(MUPulses[k] - length, 1):min(MUPulses[k] + length, len(Y))+1], np.zeros(end)))

    return MUAPs




##############################################################################################
##############################################################################################
##############################################################################################

# get the covariance matrix of the extended EMG observations

cov_mat = np.cov(np.squeeze(signal),bias=True)
# get the eigenvalues and eigenvectors of the covariance matrix
evalues, evectors  = scipy.linalg.eigh(cov_mat) 
# in MATLAB: eig(A) returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D
# sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)
sorted_evalues = np.sort(evalues)[::-1]
penalty = np.mean(sorted_evalues[len(sorted_evalues)//2:]) # int won't wokr for odd numbers
penalty = max(0, penalty)


rank_limit = np.sum(evalues > penalty)-1
if rank_limit < np.shape(signal)[0]:

    hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2

# use the rank limit to segment the eigenvalues and the eigenvectors
evectors = evectors[:,evalues > hard_limit]
evalues = evalues[evalues>hard_limit]
diag_mat = np.diag(evalues)
# np.dot is faster than @, since it's derived from C-language
# np.linalg.solve can be faster than np.linalg.inv
whitening_mat = evectors @ np.linalg.inv(np.sqrt(diag_mat)) @ np.transpose(evectors)
dewhitening_mat = evectors @ np.sqrt(diag_mat) @ np.transpose(evectors)
whitened_emg =  np.matmul(whitening_mat, signal).real 

        
init_its = np.zeros([50],dtype=int) # tracker of initialisaitons of separation vectors across iterations
fpa_its = 500 # maximum number of iterations for the fixed point algorithm
# identify the time instant at which the maximum of the squared summation of all whitened extended observation vectors
# occurs. Then, project vector is initialised to the whitened observation vector, at this located time instant.
Z = np.array(whitened_emg).copy()
sort_sq_sum_Z = np.argsort(np.square(np.sum(Z, axis = 0)))
time_axis = np.linspace(0,np.shape(Z)[1],np.shape(Z)[1])/2048  # create a time axis for spiking activity


#cf = np.square(x)
#dot_cf = 2*x
B = np.zeros([np.shape(whitened_emg)[0],50])

#################### FIXED POINT ALGORITHM #################################
init_its = sort_sq_sum_Z[-1] # since the indexing starts at -1 the other way (for ascending order list)
w_sep_vect = Z[:,int(init_its)].copy() # retrieve the corresponding signal value to initialise the separation vector
# orthogonalise separation vector before fixed point algorithm
w_sep_vect -= np.dot(np.matmul(B,B.T),w_sep_vect)
# normalise separation vector before fixed point algorithm 
w_sep_vect /= np.linalg.norm(w_sep_vect)
w_sep_vect = fixed_point_alg(w_sep_vect, B, Z)
# use the fixed point algorithm to identify consecutive separation vectors
ica_sig, spikes = get_spikes(w_sep_vect,Z)

# determine the interspike interval
ISI = np.diff(spikes/2048)
# determine the coefficient of variation
CoV = np.std(ISI)/np.mean(ISI)
# update the sepearation vector by summing all the spikes
w_n_p1 = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
# minimisation of covariance of interspike intervals
MU_filters, spikes = min_cov_isi(w_n_p1,B,Z, 2048,CoV,spikes)
B[:,0] = (w_sep_vect).real # no need to shallow copy here

# calculate SIL
fICA_source, spikes, sil = get_silohuette(MU_filters,Z,2048)

# testing peel off
fsamp = 2048
windowl = round(0.05*fsamp)
waveform = np.zeros([windowl*2+1])
firings = np.zeros([np.shape(Z)[1]])
firings[spikes] = 1 # make the firings binary
EMGtemp = np.zeros([Z.shape[0],Z.shape[1]]) # intialise a temporary EMG signal

for i in range(np.shape(Z)[0]): # iterating through the (extended) channels
    temp = cutMUAP(spikes,windowl,Z[i,:])
    waveform = np.mean(temp,axis=0)
    EMGtemp[i,:] =  scipy.signal.convolve(firings, waveform, mode = 'same',method='auto')

Z -= EMGtemp # removing the EMG representation of the source spearation vector from the signal, avoid picking up replicate content in future iterations
print('hi')