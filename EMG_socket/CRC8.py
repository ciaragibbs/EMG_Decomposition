import math
import numpy as np

def decimalToBinary(n, min_digits):
    # converting decimal to binary
    # and removing the prefix(0b)
    # add 0_padding in front until min_digits
    return str(bin(n).replace("0b", "")).zfill(min_digits)

def binaryToDecimal(n):
    return int(n, 2)

def crc_check(vector, len):
    crc = 0
    j = 0

    while len > 0:
        extract = vector[j]
        for i in range(8, 0, -1):
            # sum = xor(mod(crc, 2), mod(extract, 2));

            # ^ is xor, % is mod
            sum = int(crc % 2) ^ int(extract % 2)
            crc = math.floor(crc/2)

            if sum > 0:
                str = np.zeros([8])
                a = decimalToBinary(crc, 8)
                b = decimalToBinary(140, 8)
                for k in range(8):
                    str[k] = (not (a[k] == b[k]))

                str = str.reshape([1,8])
                index = str.shape[1]-1
                crc = 0
                for i in str[0, :]:
                    crc = crc + i * pow(2, index)
                    index = index-1

                # TODO: num2str ????
                """str = bin(str.tostring())
                crc = binaryToDecimal(str.replace(' ', ''))"""

            extract = math.floor(extract / 2)

        len = len - 1
        j = j + 1
    return crc