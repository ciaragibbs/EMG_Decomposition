
import numpy as np
import scipy
from scipy.ndimage import convolve1d
import random
from pylsl import StreamInlet, resolve_stream
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque

##################################### SYNCHRONISATION ANALYSIS ##############################################

def muscular_coherence(spikes_1,spikes_2, win_ol, win_size, no_comp, type, fsamp, its = 100):

    """ Coherence is a measure of linear correlation in the frequency domain, between two signals.
    Given posited common drive to a motor neuron pool, coherence can measure the strength of common
    input."""

    mcs =  np.empty(its) 
    counter = 0 
    used_mus_1 = set()
    used_mus_2 = set()
    # create two equally sized cumulative spike trains, calculated
    # as the sum of discharge times from 'cst_size' MUs, randomly selected, combinations are to be unique
    if type == 'inter': 
        print(' to be completed')
    elif type == 'intra':
        # spikes_1 and spikes_2 will be the same thing
        print('to be completed')
        
    while counter < its:

        # check that the 

        mus_1 = random.sample(range(np.shape(spikes_1)[0]), no_comp)
        mus_2 = random.sample(range(np.shape(spikes_2)[0]), no_comp)

        if mus_1 in used_mus_1 or mus_2 in used_mus_2:
            # we skip this iteration since this MU combination was used previously
            continue
        else:
            # we have found a unique combination of MUs relative to previous random permutations
            cst1 = np.sum(spikes_1[mus_1,:],axis=1)
            cst2 = np.sum(spikes_2[mus_2,:],axis=1)
            psd_1, faxis_1 = welch_psd(cst1,win_ol, win_size, fsamp)
            psd_2, faxis_2 = welch_psd(cst2,win_ol, win_size, fsamp)
            cpsd_12 =  welch_cpsd(cst1,cst2, win_ol, win_size, fsamp)
            # taking the magnitude of cpsd -> 
            mcs[counter] = (np.fft.fftshift(np.abs(cpsd_12)**2)) / (psd_1*psd_2) # as per Dideriksen et al. 2018  #  / win_size
            used_mus_1.add(mus_1)
            used_mus_2.add(mus_2)
            counter += 1

    return mcs, faxis_1, faxis_2

def partial_muscular_coherence():

    print('To be completed')

# Z scoring the muscular coherence values
def zscore_coherence(mcs, faxis, fsamp, seg_size, win_size, type):
    """ Assuming that the size of the data segments and window sizes are provided in seconds"""

    # In Laine et.al 2015 and Baker et.al 2003
    L = win_size / seg_size
    bias_inds = np.where((faxis >= 100) & (faxis <= 250))[0]
    mc_bias = np.sqrt(2*L)*np.arctanh(np.sqrt(mcs[bias_inds])).mean()
    mc_zscored = np.sqrt(2*L)*np.arctanh(np.sqrt(mcs)) - mc_bias
    sig_95 = np.where(mc_zscored > 1.65)[0] # used in papers, because we are doing a one sided version...?

    return mc_zscored, sig_95




# Welch's Power Spectral Density estimate
def welch_psd(signal, win_ol, win_size, fsamp):

    """ Calculates the power spectral density using a hanning window, of a user-inputted size, with 50 % overlap
    (hence Welch's method). !! this function performs a frequency shift !! """

    # typical values for win_ol are 0 and 0.5 (in terms of a fraction)
    # typical values for win_size (of the hanning window) are 0.5 - 5 seconds

    hwindow = scipy.signal.hann(win_size)
    windowed_spec = np.empty([0, win_size])
    faxis = np.arange(-np.pi, np.pi, 2*np.pi/fsamp)/(2*np.pi)*fsamp
    sig_len = np.shape(signal)[1]
    counter = 0

    # Run the loop
    while max(np.round((counter)*win_size*(1-win_ol))+np.arange(0, win_size))< sig_len:

        hanned_signal = signal[np.round((counter)*win_size*(1-win_ol))+np.arange(0, win_size)] * hwindow
        # np.fft.fftshift = shifting the zero-frequency component to the center of the spectrum
        windowed_spec[counter,:] = np.vstack((windowed_spec, np.fft.fftshift(np.abs(np.fft.fft(hanned_signal, fsamp))**2) )) #/ win_size
        counter += 1
 
    psd = windowed_spec.mean()

    return psd, faxis


# Welch's Cross Power Spectral Density estimate
def welch_cpsd(signal_1,signal_2, win_ol, win_size, fsamp):

    """ Calculates the power spectral density using a hanning window, of a user-inputted size, with 50 % overlap
    (hence Welch's method) !! this function performs no frequency shift !!  """

    # typical values for win_ol are 0 and 0.5 (in terms of a fraction)
    # typical values for win_size (of the hanning window) are 0.5 - 5 seconds

    hwindow = scipy.signal.hann(win_size)
    windowed_spec = np.empty([0, win_size])
    faxis = np.arange(-np.pi, np.pi, 2*np.pi/fsamp)/(2*np.pi)*fsamp
    sig_len = np.shape(signal_1)[1]  # for cpsd the signals should be the same size anyway
    counter = 0

    # Run the loop
    while max(np.round((counter)*win_size*(1-win_ol))+np.arange(0, win_size))< sig_len:

        hanned_signal_1 = signal_1[np.round((counter)*win_size*(1-win_ol))+np.arange(0, win_size)] * hwindow
        hanned_signal_2 = signal_2[np.round((counter)*win_size*(1-win_ol))+np.arange(0, win_size)] * hwindow
        fft_signal_1 = np.fft.fft(hanned_signal_1, fsamp)
        fft_signal_2 = np.fft.fft(hanned_signal_2, fsamp)
        # np.fft.fftshift = shifting the zero-frequency component to the center of the spectrum
        windowed_spec[counter,:] = np.vstack(fft_signal_1 * np.conj(fft_signal_2))
        counter += 1

    psd = windowed_spec.mean()

    return psd, faxis


#################################### FIRING RATE ANALYSIS #########################################

def norm_gauss_window(bin_size,win_size,std): # taken from PyalData
    # in Gallego et.al 2018
    win = scipy.signal.gaussian(int(10*std/bin_size), std/bin_size)
    return win / np.sum(win)

def half_hamming_window(l, win_size):
    # in Formento et.al 2021, 4*pi skews the window to weight latter samples with greater importance
    return 0.54 - 0.46 * np.cos(4 * np.pi * l / (win_size - 1))  

def norm_half_hamming_window(l, win_size):
    weight = 0.54 - 0.46 * np.cos(4 * np.pi * l / (win_size - 1))
    return weight / sum([half_hamming_window(i, win_size) for i in range(win_size)])

def moving_average_window(win_size):
    # in Bracklein et.al 2021
    return (1/win_size)*np.ones(win_size)

def rolling_mu_rates(signal,win_size,win_type,bin_size,method='smoothing',std=0.05):

    # signal should be noMUs x time (where this time is dependent on the buffer length)
    rate_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])

    if win_type == 'gaussian':
        win = norm_gauss_window(bin_size, std)
    elif win_type == 'ma':
        win = moving_average_window()
    elif win_type == 'half_hamming':
        win = half_hamming_window(bin_size,)

    if method == 'smoothing':
        def get_rate(spikes,win,bin_size):

            if win_type == 'gaussian' or win_type == 'ma':
                rate_signal = convolve1d(spikes, win, axis=0, output=np.float32, mode='same') / bin_size
            elif win_type == 'half_hamming':
                # TO CHECK: signal should already be equal to the window size currently...?
                rate_signal = signal * [norm_half_hamming_window(l, win_size) for l in range(win_size)]
            return rate_signal

    elif method == 'bin':
        def get_rate(spikes,win,bin_size): # was in PyalData 
            return spikes / bin_size

    # calculate rates for every spike field
    for i in range(np.shape(signal)[0]):
        rate_signal[i,:] = get_rate(signal[i,:])        

    return rate_signal


def bin_mu_spikes(signal, new_bin_size, fsamp, red_fun = np.sum, to_sqrt = 1):

    """ Assuming the bin-size is given in milli seconds """
    mus, sig_size = signal.shape
    new_bin_samps = np.round((new_bin_size/1000)*fsamp)
    T = (T // new_bin_samps) * new_bin_samps # get rid of the remaining bins at the end of the signal, that are insufficent to make a final whole bin
    signal = signal[:,:T] # clip the signal with this limit
    seg_signal = np.transpose(signal).reshape(int(T / new_bin_samps), new_bin_samps, mus) # rehsape to prepare for combining bins
    # reduction function should be np.sum if we are handling a spike signal, or np.mean if handling a rate signal
    binned_signal = red_fun(seg_signal, axis=1).squeeze().transpose()
    
    if to_sqrt:
        binned_signal = np.sqrt(binned_signal)

    return binned_signal






      
    
    





last_print = time.time()
update_interval = 125 # in ms
fsamp = 2048
averaging_factor = 8
buffer = np.zeros(fsamp * (1000/update_interval) * averaging_factor)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

channel_data = {}

for i in range(5):  # how many iterations. Eventually this would be a while True

    for i in range(16): # each of the 16 channels here
        sample, timestamp = inlet.pull_sample()
        if i not in channel_data:
            channel_data[i] = sample
        else:
            channel_data[i].append(sample)

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)


for chan in channel_data:
    plt.plot(channel_data[chan][:60])
plt.show()