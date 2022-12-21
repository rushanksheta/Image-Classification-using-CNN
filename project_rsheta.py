pipmain(['install','pandas==1.5.2'])
from pip._internal import main as pipmain
import sys
import pandas as pd
import numpy as np
from numpy import newaxis
# from sklearn.neighbors import KNeighborsClassifier
# import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, 
top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, 
ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import newaxis
from sklearn.model_selection import train_test_split
import pickle
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt
import cv2
def run(train_image, train_label, test_image):
    # Normalize and store as numpy array
    train_image = np.array(list(train_image/255))
    test_image = np.array(list(test_image/255))
    # current shape is (100000, 28, 28)
    X_train = np.array([img[:,:,newaxis] for img in train_image])
    y_train = np.array(train_label)
    # one hot encoding for train labels
    y_train = to_categorical(y_train, 100)
    # convert to shape (100000, 28, 28, 1)
    X_test = np.array([img[:,:,newaxis] for img in test_image])
    
    # convert to shape (100000, 32, 32, 1) by adding padding
    X_train_re = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test_re = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
    model = tf.keras.models.load_model('model_rsheta.h5')

    y_pred = model.predict(X_test_re)
    result = np.argmax(y_pred,axis = 1)
    
    # model = KNeighborsClassifier(n_neighbors=5)
    # data = [x.reshape(28 * 28, ) for x in train_image]
    # model.fit(data, train_label)
    # data = [x.reshape(28 * 28, ) for x in test_image]
    # result = model.predict(data)
    with open('project_xiaoq.txt', 'w') as file:  # edit here as your username
        file.write('\n'.join(map(str, result)))
        file.flush()
        return True
    return False
if __name__ == "__main__":
    # we will run your code by the following command
    # python project_xiaoq.py argv[1] argv[2]
    # argv[1] is the path of training set
    # argv[2] is the path of test set
    # for example, python demo.py train100c5k_v2.pkl test100c5k_nolabel.pkl
    try:
        df = pd.read_pickle(sys.argv[1])  # training set path
        train_data = df['data'].values
        train_target = df['target'].values
        df = pd.read_pickle(sys.argv[2])  # test set path
        test_data = df['data'].values
        info = run(train_data, train_target, test_data)
        if not info:
            print(sys.argv[0] + ": Return False")
    except RuntimeError:
        print(sys.argv[0] + ": An RuntimeError occurred")
    except:
        print(sys.argv[0] + ": An exception occurred")