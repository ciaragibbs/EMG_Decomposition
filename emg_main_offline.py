
from emg_decomposition import EMG, offline_EMG
import glob, os
import numpy as np

emg_obj = offline_EMG('/Users/cfg18/Documents/Decomposition Ciara Version/',0)
os.getcwd()
all_files = glob.glob('./*.otb+')
checkpoint = 1


for i in range(len(all_files)):

    ################## FILE ORGANISATION ################################

    emg_obj.open_otb(all_files[i]) # adds signal_dict to the emg_obj
    emg_obj.grid_formatter() # adds spatial context, and additional filtering
    
    if emg_obj.check_emg: # if you want to check the signal quality, perform channel rejection
        emg_obj.manual_rejection()

    #################### BATCHING #######################################

    if emg_obj.ref_exist: # if you want to use the target path to segment the EMG signal, to isolate the force plateau
        print('Target used for batching')
        emg_obj.batch_w_target()
    else:
        emg_obj.batch_wo_target() # if you don't have one, batch without the target path

    ################### CONVOLUTIVE SPHERING #############################
    emg_obj.signal_dict['diff_data'] = []
    tracker = 0
    nwins = int(len(emg_obj.plateau_coords)/2)
    for g in range(int(emg_obj.signal_dict['ngrids'])):

            extension_factor = int(np.round(emg_obj.ext_factor/np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]))
            # these two arrays are holding extended emg data PRIOR to the removal of edges
            emg_obj.signal_dict['extend_obvs_old'] = np.zeros([nwins, np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor), np.shape(emg_obj.signal_dict['batched_data'][tracker])[1] + extension_factor -1 - emg_obj.differential_mode ])
            emg_obj.decomp_dict['whitened_obvs_old'] = emg_obj.signal_dict['extend_obvs_old'].copy()
            # these two arrays are the square and inverse of extneded emg data PRIOR to the removal of edges
            emg_obj.signal_dict['sq_extend_obvs'] = np.zeros([nwins,np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor),np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor)])
            emg_obj.signal_dict['inv_extend_obvs'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
            # dewhitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
            emg_obj.decomp_dict['dewhiten_mat'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
            # whitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
            emg_obj.decomp_dict['whiten_mat'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
            # these two warrays are holding extended emg data AFTER the removal of edges
            emg_obj.signal_dict['extend_obvs'] = emg_obj.signal_dict['extend_obvs_old'][:,:,int(np.round(emg_obj.signal_dict['fsamp']*emg_obj.edges2remove)-1):-int(np.round(emg_obj.signal_dict['fsamp']*emg_obj.edges2remove))].copy()
            emg_obj.decomp_dict['whitened_obvs'] = emg_obj.signal_dict['extend_obvs'].copy()

            for interval in range (nwins): 
                
                # initialise zero arrays for separation matrix B and separation vectors w
                emg_obj.decomp_dict['B_sep_mat'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'][interval])[0],emg_obj.its])
                emg_obj.decomp_dict['w_sep_vect'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'][interval])[0],1])
                # MU filters needs a pre-allocation with more flexibility, to delete parts later
                emg_obj.decomp_dict['MU_filters'] = [None]*(nwins)
                #np.zeros([nwins,np.shape(emg_obj.decomp_dict['whitened_obvs'][interval])[0],emg_obj.its])
                emg_obj.decomp_dict['SILs'] = np.zeros([nwins,emg_obj.its])
                
                emg_obj.convul_sphering(g,interval,tracker)
                
    #################### FAST ICA ########################################
                emg_obj.fast_ICA_and_CKC(g,interval,tracker)

                # filtering out MUs below the SIL threshold
                #emg_obj.MUFilters[interval][:,signalprocess.SIL{nwin} < parameters.silthr] = []
                
                tracker = tracker + 1

    ##################### POSTPROCESSING #################################
            
           


