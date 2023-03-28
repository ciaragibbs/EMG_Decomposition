import glob, os, scipy
import xml.etree.ElementTree as ET
import tarfile as tf
import numpy as np
import matplotlib.pyplot as plt
from processing_tools import *
import tkinter as tk
from tkinter import simpledialog
from scipy import signal
import time
root = tk.Tk()
np.random.seed(1337)

class EMG():

    def __init__(self):
       
        self.ref_exist = 0 # if ref_signal exist ref_exist = 1; if not ref_exist = 0 and manual selection of windows
        self.windows = 3  # number of segmented windows over each contraction
        self.check_emg = 1 # 0 = Automatic selection of EMG channels (remove 5% of channels) ; 1 = Visual checking
        self.drawing_mode = 1 # 0 = Output in the command window ; 1 = Output in a figure
        self.differential_mode = 0 # 0 = no; 1 = yes (filter out the smallest MU, can improve decomposition at the highest intensities
  


#######################################################################################################
########################################## OFFLINE EMG ################################################
#######################################################################################################

class iEMG(EMG):

    # child class of EMG, so will inherit it's initialisaiton
    def __init__(self, save_dir, to_filter):

        super().__init__()
        self.save_dir = save_dir # directory at which final discharges will be saved
        self.to_filter = to_filter # whether or not you notch and butter filter the 

    
    def open_otb_multi(self, input_file):

        """ Opens data irrespective of recording modality, and parses the data array + parameters into intramusuclar versus high density surface EMG"""

        file_name = input_file.split('/')[1]
        temp_dir = os.path.join(self.save_dir, 'temp_tarholder')
        print(temp_dir)

        # make a temporary directory to store the data of the otb file if it doesn't exist yet
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

        # Open the .tar file and extract all data
        with tf.open(input_file, 'r') as emg_tar:
            emg_tar.extractall(temp_dir)

        #os.chdir(temp_dir)
        sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
        trial_label_sig = sig_files[0]  # only one .sig so can be used to get the trial name (0 index list->string)
        trial_label_xml = trial_label_sig.split('.')[0] + '.xml'
        trial_label_sig = os.path.join(temp_dir, trial_label_sig)
        trial_label_xml = os.path.join(temp_dir, trial_label_xml)

        # read the contents of the trial xml file
        with open(trial_label_xml, encoding='utf-8') as file:
            xml=ET.fromstring(file.read())

        # get sampling frequency, no. bits of AD converter, no. channels, grid names and muscle names
        fsamp = int(xml.find('.').attrib['SampleFrequency'])
        nADbit = int(xml.find('.').attrib['ad_bits'])
        nchans = int(xml.find('.').attrib['DeviceTotalChannels'])
        recording_names = [child[0].attrib['ID'] for child in xml.find('./Channels')]  # the channel description is a nested 'child' of the adapter description
        descriptions = [child[0].attrib['Description'] for child in xml.find('./Channels')]
        muscle_names = [child[0].attrib['Muscle'] for child in xml.find('./Channels')]
        todiff_inds = [int(child.attrib['ChannelStartIndex']) for child in xml.find('./Channels')]

        # take a first order difference of t

        # intialise the parsing of intramuscular and high density data
        grid_names = []
        grid_sizes = []
        grid_inds = []
        intra_names = []
        intra_sizes = []
        intra_inds = []
        nneedles = 0
        ngrids = 0

        # get the differences between the 
        recording_sizes = np.diff(todiff_inds)

        # split data into high density and intramuscular types 
        for ind, des in enumerate(descriptions):

            parsed_description = des.split()
            parsed_description =  [x.lower() for x in parsed_description]
            if 'iemg' in parsed_description:
                nneedles += 1
                intra_names.append(recording_names[ind])
                intra_sizes.append(recording_sizes[ind])
                intra_inds.append(ind)
            elif 'array' in parsed_description:
                ngrids += 1
                grid_names.append(recording_names[ind])
                grid_sizes.append(recording_sizes[ind])
                grid_inds.append(ind)


        #ngrids =int(np.floor(nchans/64))

        # read in the EMG trial data
        emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
        emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
        emg_data = emg_data.astype(float) # needed otherwise you just get an integer from the bits to microvolt division

        # convert the data from bits to microvolts
        for i in range(nchans):
            emg_data[i,:] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit))) # np.dot is faster than *

        # parse the channel data into intramuscular emg and high density surface emg
        iemg_data = np.asarray([emg_data[todiff_inds[x]:todiff_inds[x+1]] for x in intra_inds])
        semg_data = np.asarray([emg_data[todiff_inds[x]:todiff_inds[x+1]] for x in grid_inds])
       

        # squeeze/stack the data into two dimensions instead of three for easier post-processing, if the data type exists
        data_types = 'both'
        try:
            iemg_data = iemg_data.reshape((iemg_data.shape[0]*iemg_data.shape[1]), iemg_data.shape[2])
        except:
            print("This dataset does not contain intramusuclar data.")
            data_types = 'semg'
        try:
            semg_data = semg_data.reshape((semg_data.shape[0]*semg_data.shape[1]), semg_data.shape[2])
        except:
            print("This dataset does not contain surface data.")
            data_types = 'iemg'

        print(data_types)

        # creating a dictionary that separately stores data and parameters for intramuscular verrsus high density surface EMG
        if data_types == 'both':
            signal = dict(iemg_data = iemg_data, semg_data = semg_data, fsamp = fsamp, nchans = nchans, nneedles=  nneedles, intra_names = intra_names,\
                        intra_sizes = intra_sizes, ngrids = ngrids, grid_names = grid_names, grid_sizes = grid_sizes)
       
        elif data_types == 'semg':
            signal = dict(semg_data = semg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids, grid_names = grid_names, grid_sizes = grid_sizes)
        
        else: # when data_types == 'iemg':
            signal = dict(iemg_data = iemg_data, fsamp = fsamp, nchans = nchans, nneedles=  nneedles, intra_names = intra_names, intra_sizes = intra_sizes)


        # if the signals were recorded with a feedback generated by OTBiolab+, get the target and the path performed by the participant
        if self.ref_exist:

            # only opening the last two .sip files because the first is not needed for analysis
            # would only need MSE between the participant path (file 2) and the target path (file 3)
            ######## path #########
            _, target_label, path_label = glob.glob(f'{temp_dir}/*.sip')
            with open(path_label) as file:
                path = np.fromfile(file, dtype='float64')
                path = path[:np.shape(emg_data)[1]]
            ######## target ########
            with open(target_label) as file:
                target = np.fromfile(file,dtype='float64')
                target = target[:np.shape(emg_data)[1]]
            
            signal['path'] = path
            signal['target'] = target

        # delete the temp_tarholder directory since everything we need has been taken out of it
        for file_name in os.listdir(temp_dir):
            file = os.path.join(temp_dir, file_name)
            if os.path.isfile(file):
                os.remove(file)

        os.rmdir(temp_dir)
        self.signal_dict = signal
        self.decomp_dict = {} # initialising this dictionary here for later use
        # structure data from file 

        return
    

    def reject_channels_iemg(self):

        # Ask the user which channels to reject, and remove them.
        temp_intra = self.signal_dict['intra_sizes'].copy()
        temp_intra.insert(0, 0)

        for i in range(self.signal_dict['nneedles']):

            # for each needle, find how many channels there are, and plot
            # intra_sizes - for each needle, tells you how many channels there are
            sig2inspect = self.signal_dict['filtered_iemg_data'][temp_intra[i]:np.sum(temp_intra[:i+2])]
            for j in range(self.signal_dict['intra_sizes'][i]):

                num_chans2reject = []
                plt.plot(sig2inspect[j,:]/max(sig2inspect[j,:])+j+1)

            plt.show()
                
            inputchannels = simpledialog.askstring(title="Channel Rejection",
                                prompt="Please enter channel numbers to be rejected (1-indexed), input with spaces between numbers:")
            print("The selected channels for rejection are:", inputchannels)

             
            if inputchannels:
                str_chans2reject = inputchannels.split(" ")
                
                for j in range(len(str_chans2reject)):
                    num_chans2reject.append(int(str_chans2reject[j])-1)

                self.rejected_channels[i,num_chans2reject] =  1
            
            
    
    
    
    def filtering(self, emg_type = 'intra'):

        # try to filter both data types!

        try: # firstly with parameters for intramusuclar EMG
            # notch filtering
            self.signal_dict['filtered_iemg_data'] = notch_filter(self.signal_dict['iemg_data'],self.signal_dict['fsamp'])
            # bandpass filtering 
            self.signal_dict['filtered_iemg_data'] = bandpass_filter(self.signal_dict['filtered_iemg_data'],self.signal_dict['fsamp'],emg_type)      
        except:
            pass # do nothing, since we already know the data type


        try: # secondly with parameters for surface EMG
            # notch filtering 
            self.signal_dict['filtered_semg_data'] = notch_filter(self.signal_dict['semg_data'],self.signal_dict['fsamp'])
            # bandpass filtering 
            self.signal_dict['filtered_semg_data'] = bandpass_filter(self.signal_dict['filtered_semg_data'],self.signal_dict['fsamp'],'surface')  
        except:
            pass # do nothing since we already know the data type

        
    def whitening(self):

        # whitening
        # whiten the signal + impose whitened extended observation matrix has a covariance matrix equal to the identity for time lag zero i.e. Zero Phase Component Analysis (ZCA)
        
        # note NO extension is done prior to whitening
        self.decomp_dict['whitened_iemg_data'],self.decomp_dict['whiten_iemg_mat'], self.decomp_dict['dewhiten_iemg_mat'] = whiten_emg(self.signal_dict['filtered_iemg_data'])


