def notch_filter(signal,fsamp, to_han = False):

    """ Implementation of a notch filter, where the frequencies of the line interferences are unknown. Therefore, interference is defined
    as frequency components with magnitudes greater than 5 stds away from the median frequency component magnitude in a window of the signal
    - assuming you will iterate this function over each grid """

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
        """
        label_interf = np.array(label_interf,dtype=int)
        if label_interf.any():
            temp_shifted_list = np.roll(label_interf, np.arange(-int(np.floor(bandwidth_as_index / 2)), int(np.floor(bandwidth_as_index / 2) + 1)))
            interf2remove = np.where(np.isin(label_interf, temp_shifted_list, invert=True), temp_shifted_list, label_interf)
        else:
            interf2remove = label_interf"""

        if label_interf: # if a list exists
            for i in range(int(-np.floor(bandwidth_as_index/2)),int(np.floor(bandwidth_as_index/2)+1)): # so as to include np.floor(bandwidth_as_index/2)
                
                temp_shifted_list = [x + i for x in label_interf]
                interf2remove[tracker: tracker + len(label_interf)] = temp_shifted_list
                tracker = tracker + len(label_interf)
    
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        indexf2remove = np.where(np.logical_and(interf2remove >= 0 , interf2remove <= len(fourier_signal)/2))[0]
        
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        indexf2remove = np.where(np.logical_and(interf2remove >= 0 , interf2remove <= len(fourier_signal)/2))[0]
        fourier_interf[interf2remove[indexf2remove]] = fourier_signal[interf2remove[indexf2remove]] # so the interference array is non-zero where interference is identified, equal to the magnitude of interference in freq domain

        corrector = int(len(fourier_signal) - np.floor(len(fourier_signal)/2)*2)  # will either be 0 or 1 (0 if signal length is even, 1 if signal length is odd)
        # wrapping FT
        fourier_interf[int(np.ceil(len(fourier_signal)/2)):] = np.flip(np.conj(fourier_interf[1: int(np.ceil(len(fourier_signal)/2)+1- corrector)])) # not indexing first because this is 0Hz, not to be repeated
        filtered_signal[chan,:] = signal[chan,:] - np.fft.ifft(fourier_interf)
      

    return filtered_signal






##############
   def fast_ICA_and_CKC(self):

        # initialise zero arrays for separation matrix B and separation vectors w
        self.decomp_dict['B_sep_mat'] = np.zeros([np.shape(self.decomp_dict['whitened_obvs'])[0],self.its])
        self.decomp_dict['w_sep_vect'] = np.zeros([np.shape(self.decomp_dict['whitened_obvs'])[0]])
        init_its = np.zeros([self.its]) # tracker of initialisaitons of separation vectors across iterations
        fpa_its = 500 # maximum number of iterations for the fixed point algorithm
        
        # identify the time instant at which the maximum of the squared summation of all whitened extended observation vectors
        # occurs. Then, project vector is initialised to the whiteneded observation vector, at this located time instant.
        Z = self.signal_dict['whitened_obvs'] 
        sort_sq_sum_Z = np.argsort(np.square(np.sum(Z, axis = 1)))
      
        for i in range(self.its):

                #################### FIXED POINT ALGORITHM #################################
                init_its[i] = sort_sq_sum_Z[-(i+1)] # since the indexing starts at -1 the other way (for ascending order list)
                self.decomp_dict['w_sep_vect'] = Z[:,init_its[i]] # retrieve the corresponding signal value to initialise the separation vector
                # orthogonalise separation vector before fixed point algorithm
                self.decomp_dict['w_sep_vect'] = self.decomp_dict['w_sep_vect']
                # normalise separation vector before fixed point algorithm 
                self.decomp_dict['w_sep_vect'] = self.decomp_dict['w_sep_vect']/np.linalg.norm(self.decomp_dict['w_sep_vect'])
                # create a time axis for spiking activity
                time_axis = np.linspace(0,np.shape(Z)[1])/self.signal_dict['fsamp']
                # use the fixed point algorithm to identify consecutive separation vectors
                self.decomp_dict['w_sep_vect'] = fixed_point_alg(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'],Z,fpa_its)
                ### to do ### need to give option at beginning to choose contrast function, for now it is squared unless stated otherwise
                fICA_source, spikes = get_spikes(self.decomp_dict['w_sep_vect'],Z, self.signal_dict['fsamp'])

                ################# MINIMISATION OF COV OF DISCHARGES ############################
                if len(spikes) > 10:

                    # determine the interspike interval
                    ISI = np.diff(spikes/self.signal_dict['fsamp'])
                    # determine the coefficient of variation
                    CoV = np.std(ISI)/np.mean(ISI)
                    # update the sepearation vector by summing all the spikes
                    w_n_p1 = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 


                else:
                    # without enough spikes, we skip minimising the covariation of discharges to improve the separation vector
                    self.decomp_dict['B_sep_mat'][:,i] = self.decomp_dict['w_sep_vect']

            
