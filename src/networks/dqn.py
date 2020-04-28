import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from src.utils import conv2d_layer, fully_connected_layer, huber_loss
from src.networks.base import  BaseModel



#Se define la clase DQN de acuerdo con la clase BaseModel 
#para manejar el entrenamiento de la red neuronal del agente. 
class DQN(BaseModel):
    
    #Se inicializa la clase padre con el objeto config, la cantidad de acciones posibles en el ambiente,
    #el tamño del historial, el tipo de maquina en el que se entrena.
    def __init__(self, n_actions, config):
        super(DQN, self).__init__(config, "dqn")
        self.n_actions = n_actions 
        self.history_len = config.history_len
        self.cnn_format = config.cnn_format
        self.all_tf = not True


    #Se encarga de realiar un proceso de entreamiento a partir de muestra aleatoria 
    #tomada de la memoria de repetición.
    def train_on_batch_target(self, state, action, reward, state_, terminal, steps):
        state_ = state_ / 255.0  #Se acotan los valores recibidos de los estados anterior y siguiente
        state = state / 255.0
        target_val = self.q_target_out.eval({self.state_target: state_}, session=self.sess) #Calcula los retornos de ejecutar una accion deacuerdo a la red objetivo
        max_target = np.max(target_val, axis=1) #Toma el valor máximo
        target = (1. - terminal) * self.gamma * max_target + reward  #Define target con la operación. cuando terminal es true el valor es 0
        _, q, train_loss, q_summary, image_summary = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.avg_q_summary, self.merged_image_sum],
            feed_dict={
                self.state: state,
                self.action: action,
                self.target_val: target,
                self.lr: self.learning_rate
            }
        ) #Ejecuta los grafos de trainop qout loss y avgq y merged en una session de tf, isando el estad actual, las acciones y el valor de target
        if self.train_steps % 1000 == 0: #cada 1000 time steps se envía un summary para que se visualiza en tensorboard
            self.file_writer.add_summary(q_summary, self.train_steps)
            self.file_writer.add_summary(image_summary, self.train_steps)
        if steps % 20000 == 0 and steps > 50000:#se modifica la tasa deaprendizaje cada 20000 steps, después del 500000
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum: #Se revisa que no se reduzca a más del mínimo
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1#Se contabiliza el trainstep
        return q.mean(), train_loss #Se retorna el promedio de los valores q y la pérdida en ele entrenamiento.


    #Se encarga hacer un entrenamiento de la red neurona sin la red objetivo
    def train_on_batch_all_tf(self, state, action, reward, state_, terminal, steps):
        state = state/255.0 #Se acotan los valores de los estados recibidos
        state_= state_/255.0
        _, q, train_loss, q_summary, image_summary = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.avg_q_summary, self.merged_image_sum], feed_dict={
                self.state: state,
                self.action: action,
                self.state_target:state_,
                self.reward: reward,
                self.terminal: terminal,
                self.lr: self.learning_rate,
                self.dropout: self.keep_prob
            }
        ) #Ejecuta los grafos de trainop qout loss y avgq y merged en una session de tf, usando el estad actual, las acciones-
        if self.train_steps % 1000 == 0:#cada 1000 time steps se envía un summary para que se visualiza en tensorboard
            self.file_writer.add_summary(q_summary, self.train_steps)
            self.file_writer.add_summary(image_summary, self.train_steps)
        if steps % 20000 == 0 and steps > 50000:#se modifica la tasa deaprendizaje cada 20000 steps, después del 500000
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum: #Se revisa que no se reduzca a más del mínimo
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1#Se contabiliza el trainstep
        return q.mean(), train_loss #Se retorna el promedio de los valores q y la pérdida en ele entrenamiento.

    #Crea placeholders que almacenan lainformación de los pesos de las redes neuronales,
    #estados, recompensas, estado de la red n. obejetivo, la tasa de apredizaje, estados terminales,
    #valores de target.
    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.history_len, self.screen_height, self.screen_width],
                                    name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")

        self.state_target = tf.placeholder(tf.float32,
                                           shape=[None, self.history_len, self.screen_height, self.screen_width],
                                           name="input_target")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")

        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")
        self.target_val_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])

        self.learning_rate_step = tf.placeholder("int64", None, name="learning_rate_step")


    #Se encarga de realizar el proceso paso de los datos de entrenamiento a través de la red neuronal
    def add_logits_op_train(self):
        self.image_summary = []
        if self.cnn_format == "NHWC": #Se ajusta la entrada x de acuerdo lal hardware
            x = tf.transpose(self.state, [0, 2, 3, 1])
        else:
            x = self.state
        #Se lleva acabo la primera convolución sobre la enrada 
        w, b, out, summary = conv2d_layer(x, 32, [8, 8], [4, 4], scope_name="conv1_train", summary_tag="conv1_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc1"] = w #se almacena los valores de peso y bias y summary de la primera conv
        self.w["bc1"] = b
        self.image_summary.append(summary)
        #Se lleva a cabo la segunda convolución con la salida de la primera
        w, b, out, summary = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_train", summary_tag="conv2_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc2"] = w #Se almacena la valores de peso, bias y summary de la segunda conv
        self.w["bc2"] = b
        self.image_summary.append(summary)
        #Se lleva a cabo la tercerca convolución con la salida de la segunda.
        w, b, out, summary = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_train", summary_tag="conv3_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc3"] = w #Se almacena los valores de perso bias y summary de la tercera conv
        self.w["bc3"] = b
        self.image_summary.append(summary)

        shape = out.get_shape().as_list() #Se modficaa el tamaño de la salida 
        out_flat = tf.reshape(out, [-1, reduce(lambda x, y: x * y, shape[1:])])
        #Se pasa los valores a la cuarta cpa conla salida modificada de la tercerca cov
        w, b, out = fully_connected_layer(out_flat, 512, scope_name="fully1_train")

        self.w["wf1"] = w #Se almacena los valores de peso bias de la primera fullconected
        self.w["bf1"] = b
        #Se pasa los valores a la última capa 
        w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_train", activation=None)

        self.w["wout"] = w #Se almacenan los valores de peso bias para la segunda fullconected
        self.w["bout"] = b

        self.q_out = out #Se asigna el valor al atributo salida q
        self.q_action = tf.argmax(self.q_out, axis=1) #Se define la acción como el índice en que se encuentra el mayor valor en qout


    #Se encarga de realizar el proceso de entrenamiento sobre la red neuronal objetivo
    def add_logits_op_target(self):
        if self.cnn_format == "NHWC": #Se ajusta la entrada de acuerdo al hardware
            x = tf.transpose(self.state_target, [0, 2, 3, 1])
        else:
            x = self.state_target
        #Se lleva a cabo la primer convolución con la entrada x
        w, b, out, _ = conv2d_layer(x, 32, [8, 8], [4, 4], scope_name="conv1_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc1"] = w #Se almacena los valores de peso y bias
        self.w_target["bc1"] = b
        #Se lleva a cabo la segunda convolución sobre la salida de la primera conv
        w, b, out, _ = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc2"] = w#Se almacena los valores de peso y bias
        self.w_target["bc2"] = b
        #Se lleva a cabo la terce conv sobre la salida de la segunda conv
        w, b, out, _ = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc3"] = w#Se almacena los valores de peso y bias
        self.w_target["bc3"] = b

        shape = out.get_shape().as_list()  #Se modifica el tamaño de la salida de la tercer conv
        out_flat = tf.reshape(out, [-1, reduce(lambda x, y: x * y, shape[1:])])
        #Se pasa los valores a la cuarta capa 
        w, b, out = fully_connected_layer(out_flat, 512, scope_name="fully1_target")

        self.w_target["wf1"] = w#Se almacena los valores de peso y bias
        self.w_target["bf1"] = b
        #Se pasan los valores a la última capa 
        w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_target", activation=None)

        self.w_target["wout"] = w#Se almacena los valores de peso y bias
        self.w_target["bout"] = b

        self.q_target_out = out #See almacena la salida 
        self.q_target_action = tf.argmax(self.q_target_out, axis=1) #Se almacena el índice de la acción que ofrece mejor retorno

    #se encarga de iterar sobre los pesos definidos en la redes y hacer una primera actualización
    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])


    #Se encarga de calcular la pérdida del entrenamiento
    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot') #Coloca 1 en la posción correspondiente la acción
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='q_acted') #producto punto entre qout y el tensor de 1 y 0
        self.delta = train - self.target_val  #Se define valores delta
        self.loss = tf.reduce_mean(huber_loss(self.delta)) #Se calcula función de pérdida

        avg_q = tf.reduce_mean(self.q_out, 0) #Se calcula la suma de los promedio de q 
        q_summary = []
        for i in range(self.n_actions): #Se define objetos summary para ser visualizados en tensorboard
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.merged_image_sum = tf.summary.merge(self.image_summary, "images")
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    #Se encarga de cálcula la pérdida en la red neuronal objetivo
    def add_loss_op_target_tf(self):
        self.reward = tf.cast(self.reward, dtype=tf.float32)
        target_best = tf.reduce_max(self.q_target_out, 1) #Calcula los valores de q máximos 
        masked = (1.0 - self.terminal) * target_best #Anula aquellos que corresponden a estados terminales
        target = self.reward + self.gamma * masked #Calcula targe como las recompenas + el valor q máximo descontado por gamma
        ##Coloca 1 en la posción correspondiente la acción
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot') 
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1)#producto punto entre qout y el tensor de 1 y 0
        delta = target - train#Se define valores delta
        self.loss = tf.reduce_mean(huber_loss(delta)) #Se calcula función de pérdida
        avg_q = tf.reduce_mean(self.q_out, 0)#Se calcula la suma de los promedio de q 
        q_summary = []
        for i in range(self.n_actions):#Se define objetos summary para ser visualizados en tensorboard
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.merged_image_sum = tf.summary.merge(self.image_summary, "images")


    #Se encarga de crear la estructura la red neuronal
    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        if self.all_tf:
            self.add_loss_op_target_tf()
        else:
            self.add_loss_op_target()
        self.add_train_op(self.lr_method, self.lr, self.loss, clip=10)
        self.initialize_session()
        self.init_update()
    #Se encarga de actualizar los pesos de la red neuronal objetivo
    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)

