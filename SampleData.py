import numpy as np
from numpy import random
import h5py
import cv2




class DigitData(object):
    @staticmethod
    def LoadDigit():
        obj = DigitData()
        return obj.SplitData()

    def LoadRawData(self):
        h5f = h5py.File('train.h5','r')
        data = h5f['data'][:]
        label = h5f['label'][:]
        h5f.close()
        return data,label
    def SplitData(self):
        index = int(42000 - 0.1*42000)
        if index >= 42000:
            index = 41999
        data,label = self.LoadRawData()
        test_data = data[index:]
        test_label = label[index:]
        return (data,label),(test_data,test_label)
        
















