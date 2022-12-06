import numpy as np, scipy
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt

##################################### FILTERING #######################################################

def notch_filter(signal,fsamp):

    """ Implementation of a notch filter, where the frequencies of the line interferences are unknown. Therefore, interference is defined
    as frequency components with magnitudes greater than 5 stds away from the median frequency component magnitude in a window of the signal
    - assuming you will iterate this function over each grid """

    bandwidth_as_index = int(round(4*(np.shape(signal)[1]/fsamp)))
    # width of the notch filter's effect, when you intend for it to span 4Hz, but converting to indices using the frequency resolution of FT
    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])


    for chan in range(np.shape(signal)[0]):

        fourier_signal = fft(signal[chan,:]) 
        
        fourier_interf = np.zeros(len(fourier_signal),dtype = 'complex_')
        interf2remove = np.zeros(len(fourier_signal),dtype=np.int)
        window = fsamp
        tracker = 0
    
       
        for interval in range(0,len(fourier_signal),window): # so the last interval will start at len(fourier_emg) - window
            
            # range(start, stop, step)
            median_freq = np.median(abs(fourier_signal[interval+1:interval+window+1])) # so interval + 1: interval + window + 1
            std_freq = np.std(abs(fourier_signal[interval+1:interval+window+1]))
            # interference is defined as when the magnitude of a given frequency component in the fourier spectrum
            # is greater than 5 times the std, relative to the median magnitude
            label_interf = list(np.where(abs(fourier_signal[interval+1:interval+window]) > median_freq+5*std_freq)[0]) # np.where gives tuple, element access to the array
            # need to shift these labels to make sure they are not relative to the window only, but to the whole signal
            label_interf = [x + interval + 1 for x in label_interf] # + 1 since we are already related to a +1 shifted array?
    
            if label_interf: # if a list exists
                for i in range(int(-np.floor(bandwidth_as_index/2)),int(np.floor(bandwidth_as_index/2)+1)): # so as to include np.floor(bandwidth_as_index/2)
                    
                    temp_shifted_list = [x + i for x in label_interf]
                    interf2remove[tracker: tracker + len(label_interf)] = temp_shifted_list
                    tracker = tracker + len(label_interf)
        
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        indexf2remove = np.where(np.logical_and(interf2remove > 0 , interf2remove <= len(fourier_signal)/2+1))
    

        for i in indexf2remove: # maybe could be optimised to avoid repeating through indices?

            fourier_interf[interf2remove[i]] = fourier_signal[interf2remove[i]] # so the interference array is non-zero where interference is identified, equal to the magnitude of interference in freq domain

        corrector = len(fourier_signal) - np.floor(len(fourier_signal)/2)*2    # will either be 0 or 1 (0 if signal length is even, 1 if signal length is odd)
        # wrapping FT
        fourier_interf[int(np.ceil(len(fourier_signal)/2)):] = np.flip(np.conj(fourier_interf[1: int(np.ceil(len(fourier_signal)/2)+1- corrector)])) # not indexing first because this is 0Hz, not to be repeated
        filtered_signal[chan,:] = signal[chan,:] - scipy.ifft(fourier_interf)
      
    return filtered_signal

def bandpass_filter(signal,fsamp, order = 2, lowfreq = 20, highfreq = 500):

    """ Generic band-pass filter implementation and application to EMG signal 
        - assuming that you will iterate this function over each grid """
    # get the coefficients for the bandpass filter
    nyq = fsamp/2
    lowcut = lowfreq/nyq
    highcut = highfreq/nyq
    [b,a] = scipy.signal.butter(order, [lowcut,highcut],'bandpass')

    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])
    # apply filtrs
    for chan in range(np.shape(signal)[0]):
        
        filtered_signal[chan,:] = scipy.signal.filtfilt(b,a,signal[chan,:],padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return filtered_signal

################################# CONVOLUTIVE SPHERING TOOLS ##########################################################

def whiten(signal):

    print('To be completed')



###################################### FAST ICA TOOLS ##################################################################