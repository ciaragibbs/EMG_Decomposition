from iemg_decomposition import iEMG
import glob, os
import numpy as np
import matplotlib.pyplot as plt
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

iemg_obj = iEMG('/Users/cfg18/Documents/python/Intramuscular EMG/Ciara Code/pilot march 10',0)
os.getcwd()
all_files = glob.glob('./*.otb+')
i = 1
input_file =  all_files[i]


iemg_obj.open_otb(all_files[i]) # adds signal_dict to the emg_obj
iemg_obj.complete_pre_process()

print('File opening and filtering complete')
channel_data = iemg_obj.signal_dict["data"]


### plot the channels
plt.figure()
lower_limit = 200000
higher_limit = 250000
for i in  range(np.shape(channel_data)[0]):
    plt.plot(channel_data[i,lower_limit:higher_limit]/max(channel_data[i,lower_limit:higher_limit])+i+1)
plt.show()


#plt.figure()
#i = 3
#plt.plot(channel_data[i,lower_limit:higher_limit]/max(channel_data[i,lower_limit:higher_limit])+i+1)
#i = 4
#plt.plot(channel_data[i,lower_limit:higher_limit]/max(channel_data[i,lower_limit:higher_limit])+i+1)
#plt.show()



############################### SPIKE INTERFACE TESTING #################################################
"""
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp


recording = se.read_openephys('path-to-open-ephys-folder')
recording = spre.phase_shift(recording)
recording = spre.highpass_filter(recording, freq_min = 300)
recording = spre.common_reference(recording,refrence = 'median')
sorting_KS25 = ss.run_sorter('kilosort2_5',recording)
waveform_extractor = si.extract_waveforms(recording,sorting_KS25, 'wavform_folder')
qm = sqm.computer_quality_metrics(waveform_extractor)
sexp.export_to_phy(waveform_extractor,output_folder='phy_output')"""