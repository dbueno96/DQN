import numpy as np
import os
#Se deifne la clase History para complementar el proceso de entrenamiento
class History:
    #Se inicializa un tamaño de batch, un tamaño de historia y unas dimesiones de screen 
    #a partir del objeto config.
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.history_len = config.history_len
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.history = np.zeros((self.history_len, self.screen_height, self.screen_width), dtype=np.uint8)

    #Se agrea una scren al final del atributo history y se borra la primera. 
    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    #Se encarga de borrar la información de las historias
    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
