# THIS GUI IS RUN TO START AND STOP ACQUISITION, RECORDING AND DISPLAY THINGS

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



quattrocento = Amplifier()

# First the window layout in 2 columns
file_list_column = [
     # first row
    [
        sg.Button("CONNECT TO AMPLIFIER", key='connection', size=(30, 1))
    ],
    [
        sg.Text("File list"), #TEST
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"), # SEARCH BOX
        sg.FolderBrowse(), # BROWSE BUTTON
    ],

    # second row
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],

    # third row
    [
        sg.Button("START RECORDING", key='acquisition'),
        sg.Button("STOP RECORDING", key='recording'),
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:", expand_x=True, expand_y=True)],
    [sg.Text(size=(40, 1), key="-TOUT-", expand_x=True, expand_y=True)],
    [sg.Image(key="-IMAGE-", expand_x=True, expand_y=True)],

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
window = sg.Window("Interface with amplifier", layout, resizable=True, finalize=True)
window.bind('<Configure>',"Event")

acq_thread = None # acquisition thread
keylogger_thread = None # keylogger thread
# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        SocketDisconnect(quattrocento_socket) # ALWAYS DISCONNECT at the end to avoid loss of synchronization
        break
    # If folder name is filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass


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
            except:
                print('Failed to connect')

        # check connection state:  connected
        elif quattrocento.get_connection() == True and quattrocento.get_button_connection_text() =='DISCONNECT FROM AMPLIFIER':
            try:
                SocketDisconnect(quattrocento.socket) # disconnect from current socket
                quattrocento.connection_state = False # re-set connection state
                quattrocento.socket = None  # re-set socket
                quattrocento.button_connection_text = 'CONNECT TO AMPLIFIER'
                text = quattrocento.get_button_connection_text() # get new text message
                window['connection'].Update(text) # update button text
            except:
                print('Failed to disconnect')

        # sync lost... restart
        else:
            print('Please restart amplifier and program...')


    # to complete......
    elif event == 'recording':

        if acq_thread is not None and keylogger_thread is not None and quattrocento.get_acquisition() == True:
            elapsed = time.time() - t
            # STOP ACQUISITION:
            input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(stop_transfer=True)
            configuration_string = integer_to_bytes(input_command)
            SocketSend(quattrocento.socket, configuration_string)
            #elapsed = time.time() - t

            quattrocento.acquisition_state = False
            acq_thread = None

            print("Recording stopped")

    elif event == 'acquisition':

        if acq_thread is None and keylogger_thread is None and quattrocento.acquisition_state == False and quattrocento.connection_state ==True:

            # t = time.time()
            input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(stop_transfer=False)
            configuration_string = integer_to_bytes(input_command)
            SocketSend(quattrocento.socket, configuration_string)  # START ACQUISITION
            t = time.time()


            quattrocento.acquisition_state = True


            acq_thread = Thread(target=quattrocento.acquisition, args=(NrChannels, Fsampling, TransferRate, quattrocento_socket, quattrocento.get_acquisition(), t,), daemon=True)
            acq_thread.start()

        """elif acq_thread is None and quattrocento.get_acquisition() == False and quattrocento.get_connection() ==False:
            #print('Amplifier not connected... connect first')"""


window.close()
print('finished')

# ------------

"""recordings = quattrocento.acquisition_data
plt.plot(recordings[50, :])
plt.show()"""