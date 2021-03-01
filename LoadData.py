import numpy as np
import cv2
import h5py


path = "test.csv"
h5f = h5py.File('test.h5','w')

train_data = np.recfromcsv(path)
train_data = np.array(train_data)
label = []
#for tag in train_data:
#    label.append(tag[0])
data = []
for tag in train_data:
    tag = list(tag)
    data.append(tag[:])

#labelSet = h5f.create_dataset('label',data=label)

data_train = []

for i in data:
    resized = np.reshape(i,(28,28))
    data_train.append(resized)

print("Done")

dataSet = h5f.create_dataset('data',data=data_train)

h5f.close()