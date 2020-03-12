from keras.models import load_model, Sequential
# from capsnet import Capsule
from tensorflow.keras.layers import *
import cv2
from keras import activations
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
#from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x
# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)
# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).
    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )
    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """
    def __init__(self,
                 num_capsule = 20 ,
                 dim_capsule = 16,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.
        This change can improve the feature representation of Capsule.
        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])
        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)



#model = load_model('my_model.h5', custom_objects={'Capsule': Capsule(21,28,3)}, compile=False)
# model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])

# input_image = Input(shape=(None, None, 3))
# x = Conv2D(64, (3, 3), activation='relu')(input_image)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x = AveragePooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# """now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
# then connect a Capsule layer.
# the output of final model is the lengths of 10 Capsule, whose dim=16.
# the length of Capsule is the proba,
# so the problem becomes a 10 two-classification problem.
# """
# x = Reshape((-1, 128))(x)


model = load_model('my_model2.h5',custom_objects={'Capsule': Capsule,'margin_loss':margin_loss})


img = cv2.imread('pork.jpeg')
img = cv2.resize(img,(28,28))
img = img.astype('float32')
img = np.reshape(img,[1,28,28,3])
print(img)
# classes = model.predict_classes(img)
y_prob = model.predict(img) 
y_classes = y_prob.argmax(axis=-1)

print (y_classes)

data_food = pd.read_csv("Food_20new.csv") 
data_food['caption'].replace('', np.nan, inplace=True)
data_food.dropna(subset=['caption'], inplace=True)
y = data_food['caption']
# print(y)
data_food = data_food['photo_id']
le = preprocessing.LabelEncoder()
Y = le.fit_transform(y[:4677])

print(le.inverse_transform(y_classes))

loss, acc = model.evaluate(img, y_classes, verbose=2)
print(loss, acc)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))