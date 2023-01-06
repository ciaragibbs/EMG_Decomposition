import array
import socket  # for sockets
import sys  # for exit
import time  # for sleep

import numpy
import numpy as np
from pytictoc import TicToc
import matplotlib.pyplot as plt
import CRC8
import struct

# —————————————————————————–
remote_ip = "169.254.1.10"  # should match the instrument’s IP address
port = 23456  # the port number of the instrument service
CONVERSION_FACTOR = 0.000286  # conversion factor needed to get values in mV
rec = False #initialise to false
stop_transfer = False

#ConfString = [207, 9, 0, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 0, 0, 20, 5]

print("Start")


def SocketConnect(remote_ip, port):
    try:
        # create an AF_INET, STREAM socket (TCP)
        print("Create socket")
        # s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)

        sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        #sq_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


        print("created")

    except socket.error:
        print('Failed to create socket.')
        sys.exit()
    try:
        # Connect to remote server
        print("Try connection to the quattrocento")
        sq_socket.connect((remote_ip, port))
        print('connected to ip ' + remote_ip)

    except socket.error as ex:
        print('failed to connect to ip ' + remote_ip)

    return sq_socket

# Disconnect from Sessantaquattro by sending a stop command
def SocketDisconnect(conn):
    try:
        conn.shutdown(socket.SHUT_RDWR)
        """input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(stop_transfer=True)
        configuration_string = integer_to_bytes(input_command)
        SocketSend(conn, configuration_string)"""
        print('Successfully disconnected from ip ' + remote_ip)
    except:
        print('failed to disconnect from ip ' + remote_ip)



def SocketSend(quattrocento_socket, cmd):
    quattrocento_socket.send(cmd)
    return time.time()
    """response = quattrocento_socket.recv(1024)
    return response"""


# Convert integer to bytes
def integer_to_bytes(command):
    return command.to_bytes(40, byteorder="big")

def convert_bytes_to_int(bytes_value, bytes_in_sample):
    value = None
    if bytes_in_sample == 2:
        # Combine 2 bytes to a 16 bit integer value
        value = \
            bytes_value[0] * 256 + \
            bytes_value[1]
        # See if the value is negative and make the two's complement
        if value >= 32768:
            value -= 65536
    elif bytes_in_sample == 3:
        # Combine 3 bytes to a 24 bit integer value
        value = \
            bytes_value[0] * 65536 + \
            bytes_value[1] * 256 + \
            bytes_value[2]
        # See if the value is negative and make the two's complement
        if value >= 8388608:
            value -= 16777216
    else:
        raise Exception(
            "Unknown bytes_in_sample value. Got: {}, "
            "but expecting 2 or 3".format(bytes_in_sample))
    return value

# Convert channels from bytes to integers
# ORIGINAL VERSION
def bytes_to_integers(
        sample_from_channels_as_bytes,
        number_of_channels,
        bytes_in_sample,
        output_milli_volts):
    #channel_values = []


    # IMPLEMENT THIS:
    """l = time.time()
    testing = []
    for t in range(0, len(sample_from_channels_as_bytes), 2):
        testing.append(
            int.from_bytes(sample_from_channels_as_bytes[t:t + 2], byteorder='little', signed=True) * CONVERSION_FACTOR)
    print(time.time() - l)"""

    # TEST with single instant transfer:
    """channel_values = np.zeros([number_of_channels, 1])
    # Separate channels from byte-string. One channel has
    # "bytes_in_sample" many bytes in it.

    for channel_index in range(number_of_channels):
        channel_start = channel_index * bytes_in_sample
        channel_end = (channel_index + 1) * bytes_in_sample
        channel = sample_from_channels_as_bytes[channel_start:channel_end]

        # Convert channel's byte value to integer
        value = convert_bytes_to_int(channel, bytes_in_sample)
        # value = int.from_bytes(channel, byteorder='little', signed=True)

        # Convert bio measurement channels to milli volts if needed
        # The 4 last channels (Auxiliary and Accessory-channels)
        # are not to be converted to milli volts
        if output_milli_volts and channel_index < (number_of_channels - 8):
            value *= CONVERSION_FACTOR

        channel_values[channel_index, 0] = value"""
        # channel_values.append(value)

    # TEST with second by second transfer
    f_samp = int(len(sample_from_channels_as_bytes)/(number_of_channels*bytes_in_sample))
    channel_values = np.zeros([number_of_channels, f_samp])
    # Separate channels from byte-string. One channel has
    # "bytes_in_sample" many bytes in it.

    # NB: channel values are transferred in sequence --> must convert to int one by one
    # NB: the sessantaquattro uses buffer size of (bytes_in_sample * nr_channels) !!! --> doesn't multiply by f_samp
    # --> therefore for repeated only once for all channels
    # NB: the quattrocento MUST repeate the loop f_samp times too !!!
    for sample in range(f_samp):
        # print(sample)
        for channel_index in range(number_of_channels):
            # bytes_in_sample*sample*number_of_channels is to move sample by sample
            channel_start = bytes_in_sample*sample*number_of_channels + channel_index * bytes_in_sample
            channel_end = bytes_in_sample*sample*number_of_channels + (channel_index + 1) * bytes_in_sample
            channel = sample_from_channels_as_bytes[channel_start:channel_end]

            # Convert channel's byte value to integer
            #value = convert_bytes_to_int(channel, bytes_in_sample)
            value = int.from_bytes(channel, byteorder='little', signed=True)

            # Convert bio measurement channels to milli volts if needed
            # The 4 last channels (Auxiliary and Accessory-channels)
            # are not to be converted to milli volts
            if output_milli_volts and channel_index < (number_of_channels - 8):
                value *= CONVERSION_FACTOR

            channel_values[channel_index, sample] = value
            #channel_values.append(value)
    return channel_values

# NEW OPTIMISED VERSION
def bytes_to_integers_single(
        sample_from_channels_as_bytes,
        number_of_channels,
        bytes_in_sample,
        Fsampling,
        output_milli_volts):
    channel_values = []
    # Separate channels from byte-string. One channel has
    # "bytes_in_sample" many bytes in it.

    for t in range(0, len(sample_from_channels_as_bytes), bytes_in_sample):
        channel_values.append(int.from_bytes(sample_from_channels_as_bytes[t:t + bytes_in_sample], byteorder='little', signed=True) * CONVERSION_FACTOR)

    # test write directly to csv... too slow
    """for t in range(0, len(channel_values), number_of_channels):
        writer.writerow(channel_values[t:t + number_of_channels])"""

    return np.array(channel_values)

# BUFFER VERSION
def convert_from_buffer(sample_from_channels_as_bytes,
        number_of_channels,
        bytes_in_sample, Fsampling):
    b_data_int16 = np.frombuffer(buffer=sample_from_channels_as_bytes, dtype=np.int16, count=number_of_channels*Fsampling)
    b_data_uint8 = np.frombuffer(buffer=sample_from_channels_as_bytes, dtype=np.int16, count=number_of_channels*Fsampling)
    b_data_uint16 = np.frombuffer(buffer=sample_from_channels_as_bytes, dtype=np.int16, count=number_of_channels*Fsampling)
    return np.reshape(b_data_int16, [number_of_channels, Fsampling], order='c')*CONVERSION_FACTOR, np.reshape(b_data_uint8, [number_of_channels, Fsampling], order='c')*CONVERSION_FACTOR, np.reshape(b_data_uint16, [number_of_channels, Fsampling], order='c')*CONVERSION_FACTOR


def decimalToBinary(n, min_digits):
    # converting decimal to binary
    # and removing the prefix(0b)
    # add 0_padding in front until min_digits
    return str(bin(n).replace("0b", "")).zfill(min_digits)


def binaryToDecimal(n):
    return int(n, 2)




def create_bin_command_new(FSelection, Ch_sel, stop_transfer):
    NumCycles = 10 # nr times to read from 400
    PlotChan = range(16) # Channels to plot
    PlotTime = 1 # Plot time in s
    Decim = 64 # 0 (not active) or 64 (active)
    offset = 5000 # 32768
    Fsamp = [0, 8, 16, 24] # Codes to set fsamp
    FsampVal = [512, 2048, 5120, 10240] #fsamp values
    FSsel = FSelection

    if rec == True:
        acq = 1
    else:
        acq = 0

    # acq = 0 -> on
    # acq = 1 -> off
    acquisition_selection = ['10000000', '00000000']
    acquisition = acquisition_selection[acq]

    NumChan = [0, 2, 4, 6] # Codes to set nr.channels
    NumChanVal = [120, 216, 312, 408] # Nr. channels values
    NCHsel = Ch_sel # Select channel

    AnOutSource = 8 # Source input for analog output:

    AnOutChan = 0 # Channel for analog output
    AnOutGain = binaryToDecimal('00000000') # GAIN OF THE SIGNAL !!!

    GainFactor = 5 / pow(2, 16) / 150 * 1000 # Provide amplitude in mV


    AuxGainFactor = 5 / pow(2, 16) / 0.5 # Gain factor to convert Aux Channels in V

    # CREATE COMMAND:
    ConfString = np.zeros([40])
    ConfString[0] = binaryToDecimal('10000000') + Decim + Fsamp[FSsel-1] + NumChan[NCHsel-1] + 1
    ConfString[1] = AnOutGain + AnOutSource
    ConfString[2] = AnOutChan
    # -------- IN1 --------
    ConfString[3] = 0
    ConfString[4] = 0
    ConfString[5] = binaryToDecimal('00010100')
    # -------- IN2 -------- %
    ConfString[6] = 0
    ConfString[7] = 0
    ConfString[8] = binaryToDecimal('00010100')
    # -------- IN3 -------- %
    ConfString[9] = 0
    ConfString[10] = 0
    ConfString[11] = binaryToDecimal('00010100')
    # -------- IN4 -------- %
    ConfString[12] = 0
    ConfString[13] = 0
    ConfString[14] = binaryToDecimal('00010100')
    # -------- IN5 - ------- %
    ConfString[15] = 0
    ConfString[16] = 0
    ConfString[17] = binaryToDecimal('00010100')
    # -------- IN6 - ------- %
    ConfString[18] = 0
    ConfString[19] = 0
    ConfString[20] = binaryToDecimal('00010100')
    # -------- IN7 - ------- %
    ConfString[21] = 0
    ConfString[22] = 0
    ConfString[23] = binaryToDecimal('00010100')
    # -------- IN8 - ------- %
    ConfString[24] = 0
    ConfString[25] = 0
    ConfString[26] = binaryToDecimal('00010100')
    # -------- MULTIPLEIN1 - ------- %
    ConfString[27] = 0
    ConfString[28] = 0
    ConfString[29] = binaryToDecimal('00010100')
    # -------- MULTIPLEIN2 - ------- %
    ConfString[30] = 0
    ConfString[31] = 0
    ConfString[32] = binaryToDecimal('00010100')
    # -------- MULTIPLEIN3 - ------- %
    ConfString[33] = 0
    ConfString[34] = 0
    ConfString[35] = binaryToDecimal('00010100')
    # -------- MULTIPLEIN4 - ------- %
    ConfString[36] = 0
    ConfString[37] = 0
    ConfString[38] = binaryToDecimal('00010100')
    # ---------- CRC8 - --------- %
    ConfString[39] = CRC8.crc_check(ConfString, 39)

    """if rec == True:
        ConfString[0] = ConfString[0] + binaryToDecimal('00100000')  # Force the trigger to go high
        ConfString[39] = CRC8.crc_check(ConfString, 39)  # Estimates the new CRC byte"""

    if stop_transfer == True:
        ConfString[0] = binaryToDecimal('10000000') # First byte to stop transfer
        ConfString[39] = CRC8.crc_check(ConfString, 39) # Estimate new CRC

    #print(ConfString)

    NrChannels = NumChanVal[NCHsel-1]
    Fsampling = FsampVal[FSsel - 1]
    TransferRate = 2 # =16bits

    # --------------
    # --------------

    #convert to single int
    ConfString = ConfString.astype(int) # all entries to int
    ConfString_rev = ConfString[::-1] #reverse array

    digit_count = 0
    final_result = 0
    for number in ConfString_rev:
        final_result += number * pow(2, digit_count)
        #digit_count += len(str(number))
        digit_count += 8

    #print(final_result)
    # Output: bytearray(b'\x02\x03\x05\x07')

    #return ConfString, NrChannels, Fsampling, TransferRate
    return final_result, NrChannels, Fsampling, TransferRate

def empty_buffer(connection, number_of_all_channels, bytes_in_sample, Fsmapling):
    buffer_size = number_of_all_channels * bytes_in_sample * Fsmapling * 5  # send 5-sec-worth of data
    new_bytes = connection.recv(buffer_size, socket.MSG_WAITALL) # WAIT FOR ALL DATA TO BE TRANSFERRED
    return new_bytes

def last_buffer(connection, number_of_all_channels, bytes_in_sample, Fsmapling):
    buffer_size = number_of_all_channels * bytes_in_sample * Fsmapling # send 5-sec-worth of data
    new_bytes = connection.recv(buffer_size, socket.MSG_WAITALL) # WAIT FOR ALL DATA TO BE TRANSFERRED
    return new_bytes

def last_read(NrChannels, Fsampling, TransferRate, quattrocento_socket):
    rec_test = []
    recordings = np.zeros([NrChannels, 1])

    # EMPTY BUFFER AT THE END FOR THE REMAINING DATA
    sample_from_channels_as_bytes = last_buffer(
        quattrocento_socket,
        NrChannels,
        TransferRate, Fsampling)
    # print('transfer')

    sample_from_channels = bytes_to_integers(
        sample_from_channels_as_bytes,
        NrChannels,
        TransferRate,
        output_milli_volts=True)

    rec_test.append(sample_from_channels_as_bytes)
    recordings = np.concatenate((recordings, sample_from_channels), axis=1)
    recordings = np.delete(recordings, 0, 1)

    return recordings


def read_raw_bytes(connection, number_of_all_channels, bytes_in_sample, Fsmapling):
    buffer_size = number_of_all_channels * bytes_in_sample * int(Fsmapling)
    #buffer_size = number_of_all_channels * bytes_in_sample
    new_bytes = connection.recv(buffer_size, socket.MSG_WAITALL) # MSG_WAITALL ensures we have all channels everytime, even if we set Fsampling to 1 (transfer every sample)
    #new_bytes = connection.recv(buffer_size)
    return new_bytes


def check_connection(sock):
    try:
        sock.sendall(b"ping")
        return True
    except:
        return False

def connect_to_quattrocento():
    """input_command, NrChannels, Fsampling, TransferRate = create_bin_command_new(stop_transfer=False)
    configuration_string = integer_to_bytes(input_command)"""
    quattrocento_socket = SocketConnect(remote_ip, port) # CONNECT TO SOCKET
    """SocketSend(quattrocento_socket, configuration_string)"""

    return quattrocento_socket


# function to be used
def start_acquisition(NrChannels, Fsampling, TransferRate, quattrocento_socket, acq_state):
    recordings = np.zeros([NrChannels, 1])
    # recordings = []


    rec_test = []

    timing = TicToc()
    timing.tic()  # start timer
    print('START RECORDING')
    while acq_state == True:
        sample_from_channels_as_bytes = read_raw_bytes(
            quattrocento_socket,
            NrChannels,
            TransferRate, Fsampling)
        # print('transfer')

        sample_from_channels = bytes_to_integers(
            sample_from_channels_as_bytes,
            NrChannels,
            TransferRate,
            output_milli_volts=True)
        rec_test.append(sample_from_channels_as_bytes)
        # recordings.append(sample_from_channels)
        recordings = np.concatenate((recordings, sample_from_channels), axis=1)

        stop = timing.tocvalue()  # find current time
        # print(stop)

        """# TODO: INCLUDED TO STOP RUNNING AFTER N SECONDS
        if stop_acquisition == False:  # stop transmission after n seconds

            # STOP TRANSFER TO AVOID LOOSING SYNCHRONISATION WITH 400
            SocketDisconnect(quattrocento_socket)
            print("Transfer ended")
            break"""

    recordings = np.delete(recordings, 0, 1)
    return recordings

def connect_acquisition():
    configuration_string, NrChannels, Fsampling, TransferRate, quattrocento_socket = connect_to_quattrocento()
    """input_command, NrChannels, Fsampling, TransferRate  = create_bin_command_new(stop_transfer = False)
    configuration_string = integer_to_bytes(input_command)
    quattrocento_socket = SocketConnect(remote_ip, port)
    SocketSend(quattrocento_socket, configuration_string)"""

    recordings = np.zeros([NrChannels, 1])
    # recordings = []
    visualization_time = 20 #time of the visualization (in seconds) before interruption


    # turn on RECORDING
    rec = True
    """input_command, NrChannels, Fsampling, TransferRate  = create_bin_command_new()
    SocketSend(quattrocento_socket, input_command)"""

    rec_test = []

    timing = TicToc()
    timing.tic() #start timer
    print('START RECORDING')
    while True:
        sample_from_channels_as_bytes = read_raw_bytes(
            quattrocento_socket,
            NrChannels,
            TransferRate, Fsampling)
        #print('transfer')

        sample_from_channels = bytes_to_integers(
            sample_from_channels_as_bytes,
            NrChannels,
            TransferRate,
            output_milli_volts=True)
        rec_test.append(sample_from_channels_as_bytes)
        #recordings.append(sample_from_channels)
        recordings = np.concatenate((recordings, sample_from_channels), axis=1)

        # ADD REC TO SAVE DATA
        # ADD VISUALISATION

        stop = timing.tocvalue()  # find current time
        # print(stop)

        # TODO: INCLUDED TO STOP RUNNING AFTER N SECONDS
        if stop >= visualization_time:  # stop transmission after n seconds

            # STOP TRANSFER TO AVOID LOOSING SYNCHRONISATION WITH 400
            SocketDisconnect(quattrocento_socket)
            print("Transfer ended")
            break

    recordings = np.delete(recordings, 0, 1)
    return recordings

"""plt.plot(recordings[0, :])
plt.plot(recordings[5, :])
plt.plot(recordings[10, :])
plt.plot(recordings[27, :])
plt.plot(recordings[50, :])
plt.show()"""

# - CORRECT INITIAL TRANSIENT ISSUE
# -


"""- CHANGED TO MI1 AS ANALOGUE AOUTPUT
- TO CHANGE: TRY DIFFERENT GAIN
- CHECK BYTES TO INT
- conversion factor
"""