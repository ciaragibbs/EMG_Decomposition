# THIS GUI IS RUN TO START AND STOP ACQUISITION, RECORDING AND DISPLAY THINGS
import logging

import matplotlib.pyplot as plt
from quattrocento_connection import *
from amplifier_connection import *

# GUI and acquisition
import PySimpleGUI as sg
import os.path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from threading import Thread # to allow runtime of acquisition/recording and stop with button
from pytictoc import TicToc
import time
import csv

log_dir = "keylogger_files/"
log_dir_emg = "EMG_files/"

"""list_emg = []

with open('EMG_files/test.csv', newline='') as csvfile:
    emg = csv.reader(csvfile, delimiter=' ')
    for row in emg:
        list_emg.append(row)
plt.plot(list_emg[50])
plt.show()"""

quattrocento = Amplifier()


"""def save_keylogger_file(filename):
    log_dir = "keylogger_files/"
    log_dir_emg = "EMG_files/"
    #filename = input("Enter filename: ")
    path = log_dir + filename
    path2 = log_dir_emg + filename

    while os.path.isfile(path + '.csv'):  # check il that file exists already in the folder
        #filename = input("Name already used, enter new filename: ")
        window['file_text'].Update("Name already used, enter new filename: ")
        path = log_dir + filename
        path2 = log_dir_emg + filename

    print("Filename accepted, proceed with recording...")
    return path, path2"""
def refresh(window, start):
    while time.time()-start <10:
        if quattrocento.acquisition_state == True:
            window['info_text3'].Update('Recording...', text_color='red')
            window.refresh()

def plot_channels(emg, ch_range, multiple_in_nr):
    #nr_channels = len(ch_range)
    range1, range2 = ch_range[0], ch_range[1]
    nr_channels = range2 - range1
    colors = plt.rcParams["axes.prop_cycle"]()
    fig, axs = plt.subplots(nr_channels, 1, sharex=True,
                            squeeze=False)  # MUST SET SQUEEZE=FALSE !!! otherwise matlab files with only 1 channel won't work
    # setting squeeze=False forces axs to be 2D ndarray !!!

    fig.subplots_adjust(hspace=0)
    fig.suptitle("EMG data from %i channels" % nr_channels)

    t = range(emg.shape[1])

    counter = 0
    for i in range(range1, range2):
        c = next(colors)["color"]
        # axs[i].plot(t, (self.channels)[:, i], linewidth=0.5, color=c, label="Channel " + str(i+1))
        # axs[i].legend(loc="upper right")
        axs[counter, 0].plot(t, (emg)[i, :], linewidth=0.5, color=c)
        axs[counter, 0].yaxis.set_visible(False)
        counter += 1

    plt.xlabel("Time [ms]")
    # fig.text(0.04, 0.5, 'Voltage [mV]', va='center', rotation='vertical')
    #plt.title(f'Multiple-in %d', multiple_in_nr)
    plt.show()

# First the window layout in 2 columns
sg.theme("BlueMono")
sg.set_options(font=("Microsoft JhengHei", 16))

file_list_column = [
     # first row
    [
        sg.Button("CONNECT TO AMPLIFIER", key='connection', size=(30, 1))
    ],
    [
        sg.Text("File list"), #TEST
        sg.Input(key='-INPUT-', size=(20, 1), expand_x=True, expand_y=True),
        sg.FileBrowse(file_types=(("TXT Files", "*.txt"), ("ALL Files", "*.*"))),
        sg.Button("Open"),
    ],

    # second row
    [
        sg.Text("Enter filename: ", key="file_text"), #TEST
    ],

    #third row
    [
        sg.Input(key='-INPUT_filename-', size=(20, 1), expand_x=True, expand_y=True),
        sg.Button("Submit"),
    ],

    # fourth row
    [
        sg.Text("Sampling frequency (Hz):"),
        sg.Combo([512, 2048, 5120, 10240], key='combo', size=(10, 1), enable_events=True), #enable events true for returning an event
    ],

    # fifth row
    [
        sg.Text("Nr. channels:"),
        sg.Combo([120, 216, 312, 408], key='combo_2', size=(10, 1), enable_events=True),
    ],


    # sixth row
    [
        sg.Button("START RECORDING", key='acquisition'),
        sg.Button("STOP RECORDING", key='stop_recording'),
    ],

    [
        sg.Text("Amplifier not connected", key="info_text1", text_color='red'), #TEST
    ],
    [
        sg.Text("Filename not entered", key="info_text2", text_color='red'), #TEST
    ],
    [
        sg.Text("Sampling frequency not defined", key="info_text4", text_color='red'), #TEST
    ],
    [
        sg.Text("Channels not selected", key="info_text5", text_color='red', size=30), #TEST
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("", key="info_text3", text_color='red', size=30), sg.Text("", key="info_text6", text_color='red', size=30)], #TEST
    [sg.Text("TYPE HERE:", expand_x=True, expand_y=True)],
    [sg.Multiline(s=(300,10), size=(30, 10), write_only=True, expand_x=True, expand_y=True)],
    [sg.Multiline("", key='text', auto_size_text=True, size=(300, 10), expand_x=True, expand_y=True)],
    #[sg.Text("", size=(300, 10), key='text',text_color='black', background_color='white', expand_x=True, expand_y=True)],
    #[sg.Text('Enter something on Row 2'), sg.InputText()],

]

# ----- Full layout -----
layout = [
    [
        # ALLOW COLUMNS TO EXPAND
        sg.Column(file_list_column, element_justification='top'),
        sg.VSeperator(),
        sg.Column(image_viewer_column, expand_x=True, expand_y=True),
    ]
]

# TO RESIZE THE WHOLE WINDOW
#  keep_on_top=True,
window = sg.Window("Interface with amplifier", layout, resizable=True, finalize=True, use_default_focus=False, return_keyboard_events=True,)
window.bind('<Configure>',"Event")
window.bind("<Control-Alt-x>", "Control + Alt + x")
window.bind("<Control-Alt-X>", "Control + Alt + Shift + x")

window.bind("<KeyPress-a>", "a_press")
window.bind("<KeyPress-b>", "b_press")
window.bind("<KeyPress-c>", "c_press")
window.bind("<KeyPress-d>", "d_press")
window.bind("<KeyPress-e>", "e_press")
window.bind("<KeyPress-f>", "f_press")
window.bind("<KeyPress-g>", "g_press")
window.bind("<KeyPress-h>", "h_press")
window.bind("<KeyPress-i>", "i_press")
window.bind("<KeyPress-j>", "j_press")
window.bind("<KeyPress-k>", "k_press")
window.bind("<KeyPress-l>", "l_press")
window.bind("<KeyPress-m>", "m_press")
window.bind("<KeyPress-n>", "n_press")
window.bind("<KeyPress-o>", "o_press")
window.bind("<KeyPress-p>", "p_press")
window.bind("<KeyPress-q>", "q_press")
window.bind("<KeyPress-r>", "r_press")
window.bind("<KeyPress-s>", "s_press")
window.bind("<KeyPress-t>", "t_press")
window.bind("<KeyPress-u>", "u_press")
window.bind("<KeyPress-v>", "v_press")
window.bind("<KeyPress-w>", "w_press")
window.bind("<KeyPress-x>", "x_press")
window.bind("<KeyPress-y>", "y_press")
window.bind("<KeyPress-z>", "z_press")
window.bind("<KeyPress-space>", " _press")
window.bind("<KeyPress-1>", "1_press")
window.bind("<KeyPress-2>", "2_press")
window.bind("<KeyPress-3>", "3_press")
window.bind("<KeyPress-4>", "4_press")
window.bind("<KeyPress-5>", "5_press")
window.bind("<KeyPress-6>", "6_press")
window.bind("<KeyPress-7>", "7_press")
window.bind("<KeyPress-8>", "8_press")
window.bind("<KeyPress-9>", "9_press")
window.bind("<KeyPress-0>", "0_press")


acq_thread = None # acquisition thread
refresh_thread = None
count = 0
# ----------
# Run the Event Loop
while True:
    event, values = window.read()

    if quattrocento.acquisition_state==True:
        window['info_text3'].Update('Recording...', text_color='red')
        window.refresh()
    #text_elem = window['text']
    #window['info_text3'].Update('Recording...', text_color='red')


    if event in (sg.WIN_CLOSED, 'Exit'):
        SocketDisconnect(quattrocento_socket)  # ALWAYS DISCONNECT at the end to avoid loss of synchronization
        break

    # PRINTS THE LETTERS
    #sg.cprint(event, window=window, key='-ML-', c='white on purple')
    if event is not sg.TIMEOUT_KEY and quattrocento.acquisition_state==True and event != 'stop_recording':
        refresh_thread = None
        entry = [event, time.time()-quattrocento.acq_start_time]
        quattrocento.logs.append(entry)
        if len(event) == 1:
            window_text += event
            #text_elem.update(window_text)
        print('%s - %s' % (event, time.time()-quattrocento.acq_start_time))

    if event == 'combo':
       quattrocento.Fsampling = values['combo']
       window['info_text4'].Update(f'Sampling frequency: {quattrocento.Fsampling}Hz', text_color='white')

    if event == 'combo_2':
       quattrocento.channels = values['combo_2']
       MI_max = int(values['combo_2']/100) #first digit of Nr. channels
       window['info_text5'].Update(f'Channel range: MI1 - MI{MI_max} ({quattrocento.channels} channels)', text_color='white')



    elif event == 'Submit':
        filename = values['-INPUT_filename-']
        path = log_dir + filename
        path2 = log_dir_emg + filename

        if os.path.isfile(path + '.csv') or os.path.isfile(path2 + '.csv'):  # check il that file exists already in the folder
            # filename = input("Name already used, enter new filename: ")
            window['file_text'].Update("Name already used, enter new filename: ", text_color='red')
            print("Name already used, enter new filename: ")
        else:
            window['file_text'].Update("Filename accepted, proceed with recording...", text_color='white')
            print("Filename accepted, proceed with recording...")
            window['info_text2'].Update('Filename entered', text_color='white')



    elif event == 'Open':
        filename = values['-INPUT-']
        if os.path.isfile(filename):
            try:
                with open(filename) as f:
                    text = f.read()
                window['text'].update(text)
            except Exception as e:
                print("Error: ", e)


    # ----------------- CONNECTION BUTTONS

    elif event == 'connection':
        # check connection state: not connected
        if quattrocento.get_connection() == False and quattrocento.get_button_connection_text() =='CONNECT TO AMPLIFIER':
            try:
                quattrocento_socket = connect_to_quattrocento()
                quattrocento.connection_state = True # set connection state
                quattrocento.socket = quattrocento_socket # save socket
                quattrocento.button_connection_text = 'DISCONNECT FROM AMPLIFIER' #nb why it doesnt work with functions to assign !?!!?
                text = quattrocento.get_button_connection_text() # get new text message
                window['connection'].Update(text) # update button text
                window['info_text1'].Update('Amplifier connected', text_color='white')
            except:
                print('Failed to connect')

        # check connection state:  connected
        elif quattrocento.get_connection() == True and quattrocento.get_button_connection_text() =='DISCONNECT FROM AMPLIFIER':
            if quattrocento.get_acquisition() == True:
                input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(FSamp_sel, Ch_sel, stop_transfer=True)
                configuration_string = integer_to_bytes(input_command)
                t = SocketSend(quattrocento.socket, configuration_string)
            try:
                SocketDisconnect(quattrocento.socket) # disconnect from current socket
                quattrocento.connection_state = False # re-set connection state
                quattrocento.socket = None  # re-set socket
                quattrocento.button_connection_text = 'CONNECT TO AMPLIFIER'
                text = quattrocento.get_button_connection_text() # get new text message
                window['connection'].Update(text) # update button text
                window['info_text1'].Update('Amplifier not connected', text_color='red')

            except:
                print('Failed to disconnect')

        # sync lost... restart
        else:
            print('Please restart amplifier and program...')


    elif event == 'stop_recording':
        if acq_thread is not None and quattrocento.get_acquisition() == True:

            # STOP RECORDING:
            input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(FSamp_sel, Ch_sel, stop_transfer=True)
            configuration_string = integer_to_bytes(input_command)
            t = SocketSend(quattrocento.socket, configuration_string)
            quattrocento.logs.append(['_STOP_RECORDING_', t - quattrocento.acq_start_time])
            quattrocento.recording_real = t - quattrocento.acq_start_time  # total rec
            print('Recording stopped')

            window['info_text3'].Update("")
            window['info_text6'].Update("Converting and saving the files...")  # update button text

            # stack emg data sec-by-sec
            emg = np.hstack(quattrocento.recordings)


            # SAVE KEYLOGGER FILE
            with open(path + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(quattrocento.logs)

            emg = np.hstack(quattrocento.recordings)

            # SAVE EMG as PYTHON ARRAY (FAST)
            t = time.time()
            np.save(path2, emg)
            print("EMG saved: ", time.time()-t)

            # SAVE EMG - as CSV (slow)
            """with open(path2 + '.csv', 'w', newline='') as file2:
                writer = csv.writer(file2)
                writer.writerows(emg)"""


            quattrocento.acquisition_state = False
            acq_thread = None
            window['info_text6'].Update("")  # update button text


            print("Recording finished")

    elif event == 'acquisition':

        if quattrocento.channels is not None and quattrocento.Fsampling is not None and acq_thread is None and quattrocento.acquisition_state == False and quattrocento.connection_state == True and path is not None:
            quattrocento.logs = []
            window_text = ''
            quattrocento.logs.append(['EVENT', 'TIME'])

            # SETTINGS FOR THE QUATTROCENTO:
            if quattrocento.Fsampling == 512:
                FSamp_sel = 1
            elif quattrocento.Fsampling == 2048:
                FSamp_sel = 2
            elif quattrocento.Fsampling == 5120:
                FSamp_sel = 3
            elif quattrocento.Fsampling == 10240:
                FSamp_sel = 4

            if quattrocento.channels == 120:
                Ch_sel = 1
            elif quattrocento.channels == 216:
                Ch_sel = 2
            elif quattrocento.channels == 312:
                Ch_sel = 3
            elif quattrocento.channels == 408:
                Ch_sel = 4

            # START ACQUISITION (NOT RECORDING YET)
            input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(FSamp_sel, Ch_sel, stop_transfer=False)
            configuration_string = integer_to_bytes(input_command)
            SocketSend(quattrocento.socket, configuration_string)
            start = time.time()

            acq_thread = Thread(target=quattrocento.acquisition, args=(NrChannels, Fsampling, TransferRate, quattrocento_socket), daemon=True)
            acq_thread.start()
            refresh_thread = Thread(target=refresh,args=(window,start,), daemon=True)
            refresh_thread.start()


        """elif acq_thread is None and quattrocento.get_acquisition() == False and quattrocento.get_connection() ==False:
            #print('Amplifier not connected... connect first')"""

window.close()


# ------------
print('Transfer ended')
"""recordings = quattrocento.recordings
plt.plot(recordings[50, :])
plt.show()"""

#RMS
"""rms = np.sqrt(np.average(emg[130:380, 0:7500]**2, axis=1))*1000"""

# slow 90
# medium 110
# fast normal


"""recordings = quattrocento.recordings
plt.plot(recordings[40, :])
plt.show()

recordings = quattrocento.recordings
plt.plot(recordings[150, :])
plt.show()

recordings = quattrocento.recordings
plt.plot(recordings[260, :])
plt.show()

recordings = quattrocento.recordings
plt.plot(recordings[360, :])
plt.show()"""


# gui con channels, frequency, filtering, bigger font size, empty buffer, savoing, converting...
# connverter bytes
# channel plotting
# emg + lettere overlaid
# nuovi dialoghi, metronomo, numeri, parole, impostazioni seduta
# planning report + immagini + bullet points
# aggiornamento elettrodi
# poggia polso
