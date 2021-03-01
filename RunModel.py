from keras import backend as K
from keras.datasets import mnist
from keras.models import model_from_json
from keras.optimizers import SGD
import h5py
import scipy.misc
import numpy as np
import cv2



def LoadData():
    h5f = h5py.File('test.h5','r')
    data = h5f['data'][:]
    #label = h5f['label'][:]
    h5f.close()
    return data




X_test = LoadData()
K.set_image_data_format('channels_first')







model_architecture = 'digit_config.json'
model_weights = 'digit_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)


optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
	metrics=['accuracy'])


index = 1
submit = open('submission.csv',mode='w')


for chuso in X_test:
    chuso = np.array(chuso).astype(np.uint8)
    chuso = np.array([[chuso]]) 
    predictions = model.predict(chuso)
    result = np.argmax(predictions)
    submit.write(str(index)+','+str(result)+'\n')
    index = index + 1

submit.close()






