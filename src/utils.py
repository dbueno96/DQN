from scipy.misc import imresize as resize
import json
import numpy as np
import tensorflow as tf

#Se encarga de realizar el producto punto para obtener el color deseadp
def rgb2gray(screen):
    return np.dot(screen[..., :3], [0.299, 0.587, 0.114])


def load_config(config_file):
    pass

#Se encarga de guardar un archivo json
def save_config(config_file, config_dict):
    with open(config_file, 'w') as fp:
        json.dump(config_dict, fp)

#Define una capa convolucionalar y su compartamiento dependiente de si se ejecuta el 
#algoritmo en gpu o cpu
def conv2d_layer(x, output_dim, kernel_size, stride, initializer=None, padding="VALID", data_format="NCHW",
                 summary_tag=None,
                 scope_name="conv2d", activation=tf.nn.relu):
    with tf.variable_scope(scope_name):
        if data_format == 'NCHW': #Define el paso y tamaños iniciales para CPU
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC': #Define el paso y tamaños inciales para gpu
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=tf.truncated_normal_initializer(0, 0.02)) #Define los pesos variables de la capa
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0)) #Define el error como variable
        out = tf.nn.bias_add(conv, b, data_format) #Define la salida como la sima entre conv y bias
        
        if activation != None: #si hay alguna activación
            out = activation(out) #se aplica la activación sobre la salida
        summary = None
        if summary_tag is not None: #Define valores de dimensiones de salida.
            # TODO general definitions
            if output_dim == 32:
                ix = 4
                iy = 8
            elif output_dim == 64:
                ix = 8
                iy = 8

            img = tf.slice(out, [0, 0, 0, 0], [1, -1, -1, -1]) #Define un tamaño de img a partir de la salida
            if data_format == "NCHW": #Si se ejecuta en gpu se aplica una transposición a la imagen  y permuta las posicones
                img = tf.transpose(img, [0, 2, 3, 1])
            out_shape = img.get_shape().as_list()  #define la forma de la salida a partir la imagen
            img = tf.reshape(img, [out_shape[1], out_shape[2], out_shape[3]]) #Modifica el tamo de img de acuerdo a la forma de salida
            out_shape[1] += 4
            out_shape[2] += 4
            img = tf.image.resize_image_with_crop_or_pad(img, out_shape[1], out_shape[2]) #Cambia el tamaño de la imagen
            img = tf.reshape(img, [out_shape[1], out_shape[2], ix, iy])  
            img = tf.transpose(img, [2, 0, 3, 1])
            img = tf.reshape(img, [1, ix * out_shape[1], iy * out_shape[2], 1])
            summary = tf.summary.image(summary_tag, img)
        return w, b, out, summary #Se retornan los pesos el error, la salida y la summary


#Define una capa completamente conectada
def fully_connected_layer(x, output_dim, scope_name="fully", initializer=tf.random_normal_initializer(stddev=0.02),
                          activation=tf.nn.relu):
    shape = x.get_shape().as_list() #define los tamaños manejados en la capa
    with tf.variable_scope(scope_name):
        w = tf.get_variable("w", [shape[1], output_dim], dtype=tf.float32,
                            initializer=initializer) #Define los pesos variables, de acuerdo al tamño de entrada y salida
        b = tf.get_variable("b", [output_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer()) #Define los errores bariables de acuerdo a al tamaño de salida
        out = tf.nn.xw_plus_b(x, w, b) #Define la salida como la suma  xw+b
        if activation is not None: # Si hay alguna función de activación se aplica sobre out
            out = activation(out)

        return w, b, out

def stateful_lstm(x, num_layers, lstm_size, state_input, scope_name="lstm"):
    with tf.variable_scope(scope_name):
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cell, x, initial_state=state_input)
        return outputs, state

#Se aplica 0.5 x^2 en los valores de x que cumplen que |x| < delta. 
#En los que no, se aplica delta * |x| - 0.5*delta
def huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), delta * tf.abs(x) - 0.5* delta)

#Se calcula la productoria de los elementos x, y retorna el entero más cercano.
def integer_product(x):
    return int(np.prod(x))


def initializer_bounds_filter(filter_shape):
    fan_in = integer_product(filter_shape[:3])
    fan_out = integer_product(filter_shape[:2]) * filter_shape[3]
    return np.sqrt(6. / (fan_in + fan_out))
