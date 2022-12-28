import numpy as np
import scipy
from scipy import signal
from numpy import linalg
from scipy.fft import fft
import matplotlib.pyplot as plt
import sklearn 
from sklearn.cluster import KMeans

##################################### FILTERING TOOLS #######################################################

def notch_filter(signal,fsamp,to_han = False):

    """ Implementation of a notch filter, where the frequencies of the line interferences are unknown. Therefore, interference is defined
    as frequency components with magnitudes greater than 5 stds away from the median frequency component magnitude in a window of the signal
    - assuming you will iterate this function over each grid 
    
    IMPORTANT!!! There is a small difference in the output of this function and that in MATLAB, best guess currently is differences in inbuilt FFT implementation."""

    bandwidth_as_index = int(round(4*(np.shape(signal)[1]/fsamp)))
    # width of the notch filter's effect, when you intend for it to span 4Hz, but converting to indices using the frequency resolution of FT
    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])


    for chan in range(np.shape(signal)[0]):

        if to_han:
            hwindow = scipy.signal.hann(np.shape(signal[chan,:])[0])
            final_signal = signal[chan,:]* hwindow
        else:
            final_signal = signal[chan,:]

        fourier_signal = np.fft.fft(final_signal)
        fourier_interf = np.zeros(len(fourier_signal),dtype = 'complex_')
        interf2remove = np.zeros(len(fourier_signal),dtype=np.int)
        window = fsamp
        tracker = 0
    
        for interval in range(0,len(fourier_signal)-window+ 1,window): # so the last interval will start at len(fourier_emg) - window
            
            # range(start, stop, step)
            median_freq = np.median(abs(fourier_signal[interval+1:interval+window+1])) # so interval + 1: interval + window + 1
            std_freq = np.std(abs(fourier_signal[interval+1:interval+window+1]))
            # interference is defined as when the magnitude of a given frequency component in the fourier spectrum
            # is greater than 5 times the std, relative to the median magnitude
            label_interf = list(np.where(abs(fourier_signal[interval+1:interval+window+1]) > median_freq+5*std_freq)[0]) # np.where gives tuple, element access to the array
            # need to shift these labels to make sure they are not relative to the window only, but to the whole signal
            label_interf = [x + interval + 1 for x in label_interf] # + 1 since we are already related to a +1 shifted array?
    
            if label_interf: # if a list exists
                for i in range(int(-np.floor(bandwidth_as_index/2)),int(np.floor(bandwidth_as_index/2)+1)): # so as to include np.floor(bandwidth_as_index/2)
                    
                    temp_shifted_list = [x + i for x in label_interf]
                    interf2remove[tracker: tracker + len(label_interf)] = temp_shifted_list
                    tracker = tracker + len(label_interf)
        
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        indexf2remove = np.where(np.logical_and(interf2remove >= 0 , interf2remove <= len(fourier_signal)/2))[0]
        fourier_interf[interf2remove[indexf2remove]] = fourier_signal[interf2remove[indexf2remove]]
        corrector = int(len(fourier_signal) - np.floor(len(fourier_signal)/2)*2)  # will either be 0 or 1 (0 if signal length is even, 1 if signal length is odd)
        # wrapping FT
        fourier_interf[int(np.ceil(len(fourier_signal)/2)):] = np.flip(np.conj(fourier_interf[1: int(np.ceil(len(fourier_signal)/2)+1- corrector)])) # not indexing first because this is 0Hz, not to be repeated
        filtered_signal[chan,:] = signal[chan,:] - np.fft.ifft(fourier_interf)
      

    return filtered_signal


   

def bandpass_filter(signal,fsamp, order = 2, lowfreq = 20, highfreq = 500):

    """ Generic band-pass filter implementation and application to EMG signal  - assuming that you will iterate this function over each grid """

    """IMPORTANT!!! There is a difference in the default padding length between Python and MATLAB. For MATLAB -> 3*(max(len(a), len(b)) - 1),
    for Python scipy -> 3*max(len(a), len(b)). So I manually adjusted the Python filtfilt to pad by the same amount as in MATLAB, if you don't the results will not match across
    lanugages. """   

    # get the coefficients for the bandpass filter
    nyq = fsamp/2
    lowcut = lowfreq/nyq
    highcut = highfreq/nyq
    [b,a] = scipy.signal.butter(order, [lowcut,highcut],'bandpass') # the cut off frequencies should be inputted as normalised angular frequencies

    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])
    # construct and apply filter
    for chan in range(np.shape(signal)[0]):
        
        filtered_signal[chan,:] = scipy.signal.filtfilt(b,a,signal[chan,:],padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return filtered_signal

################################# CONVOLUTIVE SPHERING TOOLS ##########################################################

def extend_emg(extended_template, signal, ext_factor):

    """ Extension of EMG signals, for a given window, and a given grid. For extension, R-1 versions of the original data are stacked, with R-1 timeshifts.
    Structure: [channel1(k), channel2(k),..., channelm(k); channel1(k-1), channel2(k-1),...,channelm(k-1);...;channel1(k - (R-1)),channel2(k-(R-1)), channelm(k-(R-1))] """

    # signal = self.signal_dict['batched_data'][tracker][0:] (shape is channels x temporal observations)

    nchans = np.shape(signal)[0]
    nobvs = np.shape(signal)[1]
    for i in range(ext_factor):

        lag_block = i + 1
        extended_template[nchans*(lag_block-1):nchans*lag_block, i:nobvs +i] = signal
  
    return extended_template


def whiten_emg(signal):
    
    """ Whitening the EMG signal imposes a signal covariance matrix equal to the identity matrix at time lag zero. Use to shrink large directions of variance
    and expand small directions of variance in the dataset. With this, you decorrelate the data. """

    # get the covariance matrix of the extended EMG observations
    
    cov_mat = np.cov(np.squeeze(signal))
    # get the eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors  = scipy.linalg.eig(cov_mat) 
    # in MATLAB: eig(A) returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D
    # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)
    sorted_evalues = np.flip(np.sort(evalues))
    penalty = np.mean(sorted_evalues[int(len(sorted_evalues)/2):])
  
    if penalty < 0:
        penalty = 0

    rank_limit = np.sum(evalues > penalty)-1
    if rank_limit < np.shape(signal)[0]:

        hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2
    
    # use the rank limit to segment the eigenvalues and the eigenvectors
    evectors = evectors[:,evalues > hard_limit]
    evalues = evalues[evalues>hard_limit]
    diag_mat = np.diag(evalues)
    
    
    whitening_mat = evectors @ np.linalg.inv(np.sqrt(diag_mat)) @ np.transpose(evectors)
    dewhitening_mat = evectors @ np.sqrt(diag_mat) @ np.transpose(evectors)
    whitened_emg =  whitening_mat @ signal
    
    return whitened_emg









###################################### DECOMPOSITION TOOLS ##################################################################

def square(x):
    return np.square(x)

def skew(x):
    return x**3/3

def exp(x):
    return np.exp(-x**2/2)

def logcosh(x):
    return np.log(np.cosh(x))

def dot_square(x):
    return 2*x

def dot_skew(x):
    return 2*(x**2)/3

def dot_exp(x):
    return -1*(np.exp(-np.square(x)/2)) + (np.square(x)) @ np.exp(-np.square(x)/2)

def dot_logcosh(x):
    return np.tanh(x)

def fixed_point_alg(w_n, B, Z, its=500, cf='square'):
    
    if cf == 'square':
        cf = square
        dot_cf = dot_square
    elif cf == 'skew':
        cf = skew
        dot_cf = dot_skew
    elif cf == 'exp':
        cf = exp
        dot_cf = dot_exp
    elif cf == 'logcosh':
        cf = logcosh
        dot_cf = dot_logcosh

    w_n = np.expand_dims(w_n, axis=1)
    B_T_B = B @ np.transpose(B)
    its_tolerance_sq = its_tolerance ** 2

    while True:
        w_n_1 = w_n
        A = dot_cf(np.dot(np.transpose(w_n_1), Z)).mean()
        w_n = np.expand_dims((Z * cf(np.dot(np.transpose(w_n_1), Z))).mean(axis=1), axis=1) - A * w_n_1
        w_n = w_n - B_T_B @ w_n
        if (np.square(w_n - w_n_1)).sum() < its_tolerance_sq:
            break

    return w_n

def fixed_point_alg(w_n, B, Z, its = 500, cf = 'square'):

    """ Update function for source separation vectors. The code user can select their preferred contrast function using a string input:
    1) square --> x^2
    2) logcosh --> log(cosh(x))
    3) exp  --> e(-x^2/2)
    4) skew --> -x^3/3 
    e.g. skew is faster, but less robust to outliers relative to logcosh and exp
    Upon meeting a threshold difference between iterations of the algorithm, separation vectors are discovered 
    
    The maximum number of iterations (its) and the contrast function type (cf) are already specified, unless alternative input is provided. """


    if cf == 'square':
        cf = lambda x: np.square(x)
        dot_cf = lambda x: 2*x
    elif cf == 'skew':
        cf = lambda x: x**3/3
        dot_cf = lambda x: 2*(x**2)/3
    elif cf == 'exp':
        cf = lambda x: np.exp(-x**2/2)
        dot_cf = lambda x: -1*(np.exp(-np.square(x)/2)) + (np.square(x)) @ np.exp(-np.square(x)/2)
    elif cf == 'logcosh':
        cf = lambda x: np.log(np.cosh(x))
        dot_cf = lambda x: np.tanh(x)

    
   
    counter = 0
    its_tolerance = 0.0001
    sep_diff = np.ones([its])
    w_n = np.expand_dims(w_n,axis=1)

    while sep_diff[counter] > its_tolerance:

        # transfer current separation vector as the previous arising separation vector
        w_n_1 = w_n 
        # use update function to get new, current separation vector
        #A = np.mean(dot_cf(np.transpose(w_n_1) @ Z))
        A =  dot_cf(np.dot(np.transpose(w_n_1), Z)).mean()
        w_n = np.expand_dims((Z*cf(np.dot(np.transpose(w_n_1), Z))).mean(axis=1),axis=1) - A * w_n_1
        # orthogonalise separation vectors
        w_n = w_n - B @ np.transpose(B)@ w_n
        # normalise separation vectors
        w_n = w_n/np.linalg.norm(w_n)
        counter = counter + 1
        sep_diff[counter] = abs(np.transpose(w_n) @ w_n_1 - 1)

    return w_n



def get_spikes(w_n,Z, fsamp):

    """ Based on gradient convolutive kernel compensation. Aim to remove spurious discharges to improve the source separation
    vector estimate. Results in a reduction in ISI vairability (by seeking to minimise the covariation in MU discharges)"""

    # Step 4a: 
    source_pred = np.square((np.transpose(w_n) @ Z)).real # element-wise square of the input to estimate the ith source
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = fsamp*0.01 ) # peaks variable holds the indices of all peaks
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[:,peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes = peaks[kmeans.labels_ == spikes_ind]

    return source_pred, spikes

def min_cov_isi(w_n,B,Z,fsamp,cov_n,spikes_n):  #

    " Iteratively computing the "

    counter = 0
    cov_n_1 = 2 * cov_n
    while cov_n < cov_n_1:

        cov_n_1 = cov_n
        spikes_n_1 = spikes_n
        w_n = np.expand_dims(w_n,axis=1)
        w_n_1 = w_n
        _ , spikes = get_spikes(w_n,Z,fsamp)

        ################# MINIMISATION OF COV OF DISCHARGES ############################
        if len(spikes) > 10: # why in MATLAB code this threshold is removed for the minimisation loop?

            # determine the interspike interval
            ISI = np.diff(spikes/fsamp)
            # determine the coefficient of variation
            CoV = np.std(ISI)/np.mean(ISI)
            # update the sepearation vector by summing all the spikes
            w_n = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
            counter = counter + 1

    # if you meet the CoV minimisation condition, but with a single-spike-long train, use the updated
    # separation vector and recompute the spikes

    if len(spikes) < 2:
        _ , spikes = get_spikes(w_n,Z,fsamp)

    return np.squeeze(w_n_1), spikes_n_1


################################ VALIDATION TOOLS ########################################

def get_silohuette(w_n,Z,fsamp):

    # Step 4a: 
    source_pred = (np.transpose(w_n) @ Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.01) ) # this is approx a value of 20, which is in time approx 10ms
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes = peaks[kmeans.labels_ == spikes_ind]
        sil = sklearn.metrics.silhouette_score(source_pred[peaks].reshape(-1,1), kmeans.labels_, metric='euclidean')
    else:

        sil = 0

    return source_pred, spikes, sil

def peel_off(Z,spikes,fsamp):

    windowl = round(0.05*fsamp)
    waveform = np.zeros([windowl*2+1])
    firings = np.zeros([np.shape(Z)[1]])
    firings[spikes] = 1; # make the firings binary
    EMGtemp = np.zeros([np.shape(Z)[0],np.shape(Z)[1]]); # intialise a temporary EMG signal

    for i in range(np.shape(Z)[0]): # iterating through the (extended) channels
        temp = cutMUAP(spikes,windowl,Z[i,:])
        waveform = np.mean(temp,axis=0)
        EMGtemp[i,:] =  np.convolve(firings, waveform, "same")

    Z = Z - EMGtemp; # removing the EMG representation of the source spearation vector from the signal, avoid picking up replicate content in future iterations

############################## POST PROCESSING #####################################################

def cutMUAP(MUPulses, length, Y):

    """ Direct converion of MATLAB code in-lab. Extracts consecutive MUAPs out of signal Y and stores
    them row-wise in the out put variable MUAPs.
    Inputs: 
    - MUPulses: Trigger positions (in samples) for rectangualr window used in extraction of MUAPs
    - length: radius of rectangular window (window length = 2*len +1)
    - Y: Single signal channel (raw vector containing a single channel of a recorded signals)
    Outputs:
    - MUAPs: row-wise matrix of extracted MUAPs (algined signal intervals of length 2*len+1)"""
 
    while len(MUPulses) > 0 and MUPulses[len(MUPulses)-1] + 2 * length > len(Y):
        MUPulses = MUPulses[:-1]

    c = len(MUPulses)
    edge_len = round(length / 2)
    tmp = scipy.signal.windows.gaussian(2 * edge_len, std=edge_len / 3)
    win = np.concatenate((tmp[:edge_len-1], np.ones(2 * length - 2 * edge_len + 1), tmp[edge_len-1:]))
    MUAPs = np.zeros((c, 1 + 2 * length))
    for k in range(c):
        start = max(MUPulses[k] - length-1, 1) - (MUPulses[k] - length-1)
        end = MUPulses[k] + length - 1 - min(MUPulses[k] + length - 1, len(Y))
        MUAPs[k, :] = win * np.concatenate((np.zeros(start), Y[max(MUPulses[k] - length - 1, 1):min(MUPulses[k] + length - 1, len(Y))], np.zeros(end)))

    return MUAPs

   