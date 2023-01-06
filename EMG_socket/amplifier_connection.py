import numpy as np
from quattrocento_connection import *
import time


class Amplifier:
    # here data is given as obj constructor
    def __init__(self): # todo: CORRECT BY PUTTING CHANNELS INTO EACH FUNCTION !
        self.acquisition_state = False
        self.recording_state = False
        self.connection_state = False
        self.socket = None
        self.remote_ip = "169.254.1.10"  # should match the instrument’s IP address
        self.port = 23456  # the port number of the instrument service
        self.CONVERSION_FACTOR = 0.000286
        self.acquisition_data = None
        self.recordings = None
        self.rec_test = None
        self.acq_start_time = 0
        self.acq_real_time = 0
        self.recording_length = 0
        self.recording_real = 0
        self.logs = []
        self.Fsampling = None
        self.channels = None

        self.button_connection_text = 'CONNECT TO AMPLIFIER'

    def acquisition(self, NrChannels, Fsampling, TransferRate, quattrocento_socket):

        self.recordings = []
        self.rec_test = []

        print('EMPTYING THE BUFFER...')
        # EMPTY THE BUFFER TO AVOID INITIAL DELAY:
        empty_buffer(
            quattrocento_socket,
            NrChannels,
            TransferRate, Fsampling)

        print('Acquiring emg...')
        print('START TYPING: \n')
        self.acquisition_state = True
        self.acq_start_time = time.time()

        self.logs.append(['_START_RECORDING_', 0]) # Initialise keylogger almost together as recording
        while self.acquisition_state==True:
            #t = time.time()
            sample_from_channels_as_bytes = read_raw_bytes(
                quattrocento_socket,
                NrChannels,
                TransferRate, Fsampling)

            self.rec_test.append(sample_from_channels_as_bytes)

            #t = time.time()
            sample_from_channels = bytes_to_integers_single(
                sample_from_channels_as_bytes,
                NrChannels,
                TransferRate,
                Fsampling,
                output_milli_volts=True)

            self.recordings.append(sample_from_channels.reshape(-1, NrChannels).transpose())
            #print(time.time()-t)


    def recording(self, state):
        self.recording = state

    def connection(self, state):
        self.connection = state
        self.button_connection_text = 'DISCONNECT FROM AMPLIFIER'

    def socket(self, sock):
        self.socket = sock

    def get_connection(self):
        return self.connection_state

    def get_acquisition(self):
        return self.acquisition_state

    def get_recording(self):
        return self.recording_state

    def get_button_connection_text(self):
        return self.button_connection_text

    def get_socket(self):
        return self.socket()
