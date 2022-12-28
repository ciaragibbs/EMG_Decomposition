from emg_decomposition_off import EMG, preprocess_EMG
import glob, os
import numpy as np

emg_obj = preprocess_EMG()
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
    for g in range(int(emg_obj.signal_dict['ngrids'])):

            emg_obj.signal_dict['extend_obvs'] = []
            emg_obj.signal_dict['sq_extend_obvs'] = []
            emg_obj.signal_dict['inv_extend_obvs'] = []
            emg_obj.decomp_dict['whitened_obvs'] = []

            for interval in range (int(len(emg_obj.plateau_coords)/2)):
                emg_obj.convul_sphering(g,interval,tracker)
                
    #################### FAST ICA ########################################
                emg_obj.fast_ICA_and_CKC(g,interval,tracker)
                
                tracker = tracker + 1



