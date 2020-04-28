from src.env_wrapper import GymWrapper
import numpy as np
#Se define la base sobre la que se fundamenta un agente en esta implementación
#a partir de la información de un objeto config. 
class BaseAgent():
    # Inicializa una copia del objeto config, un GymWrapper para el ambiente de gym,
    # una recompensa, una longito, un valor epsilon, una recompensa máxima y mínima,
    # una memoria de repetición, un historial y una red neuronal
    def __init__(self, config):
        self.config = config
        self.env_wrapper = GymWrapper(config)
        self.rewards = 0
        self.lens = 0
        self.epsilon = config.epsilon_start
        self.min_reward = -1.
        self.max_reward = 1.0
        self.replay_memory = None
        self.history = None
        self.net = None
        if self.config.restore:
            self.load()
        else:
            self.i = 0


    #Se encarga de guardar un checkpoint del entrenamiento de la red neuronal y 
    #de la memoria de repetición.
    def save(self):
        self.replay_memory.save()
        self.net.save_session()
        np.save(self.config.dir_save+'step.npy', self.i)
    #Se encarga de leer la información de la memoria de repetición y del entrenamiento de la 
    #red neuronal a partir de las rutas definidas para ello en el atributo config.
    def load(self):
        self.replay_memory.load()
        self.net.restore_session()
        self.i = np.load(self.config.dir_save+'step.npy')

