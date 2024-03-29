from tensorflow.python.client import device_lib
#Se encarga de verificar la disponibilidad de GPU y usarlas para el entrenamiento.
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#La clase config contiene la mayoría de los atributos que se necesitan para 
#para la correcta ejecución del las otras clases.
class Config(object):

    train_steps = 50000000
    batch_size = 64
    history_len = 4
    frame_skip = 4
    epsilon_start = 1.0
    epsilon_end = 0.02
    max_steps = 10000
    epsilon_decay_episodes = 1000000
    train_freq = 8
    update_freq = 10000
    train_start = 20000
    dir_save = "saved_session/"
    restore = False
    epsilon_decay = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
    random_start = 10
    test_step = 5000
    network_type = "dqn"


    gamma = 0.99
    learning_rate_minimum = 0.00025
    lr_method = "rmsprop"
    learning_rate = 0.00025
    lr_decay = 0.97
    keep_prob = 0.8

    num_lstm_layers = 1
    lstm_size = 512
    min_history = 4
    states_to_update = 4

    if get_available_gpus():
        cnn_format = "NCHW"
    else:
        cnn_format = "NHWC"


#Se define la clas GymConfig para definir algunos atributos necesario para 
#la creación del ambiente gym posterioemente
class GymConfig(Config):
    state = None
    screen_height = 84
    screen_width = 84
    env_name = "Breakout-v0"
    mem_size = 800000


