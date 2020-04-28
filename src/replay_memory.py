import numpy as np
import random
import os


# Define la base de la clase ReplayMemoria. Inicializa un copia del objeto config
# un arreglo de acciones y otro de recompensas, un arreglo de pantallans y terminales con
# con tamaños definidos en config. Un contadores y valor ctual y un directorio de guardado
class ReplayMemory:

    def __init__(self, config):
        self.config = config
        self.actions = np.empty((self.config.mem_size), dtype=np.int32)
        self.rewards = np.empty((self.config.mem_size), dtype=np.int32)
        # Screens are dtype=np.uint8 which saves massive amounts of memory, however the network expects state inputs
        # to be dtype=np.float32. Remember this every time you feed something into the network
        self.screens = np.empty((self.config.mem_size, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.terminals = np.empty((self.config.mem_size,), dtype=np.float16)
        self.count = 0
        self.current = 0
        self.dir_save = config.dir_save + "memory/"

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
    #Se encarga de guardar los arreglos numpy en el directorio definido para ello. 
    def save(self):
        np.save(self.dir_save + "screens.npy", self.screens)
        np.save(self.dir_save + "actions.npy", self.actions)
        np.save(self.dir_save + "rewards.npy", self.rewards)
        np.save(self.dir_save + "terminals.npy", self.terminals)

    #Se encarga de cargar los arreglo numpy desde archivos npy y guardarlos en los arreglos destinados para estos valores.
    def load(self):
        self.screens = np.load(self.dir_save + "screens.npy")
        self.actions = np.load(self.dir_save + "actions.npy")
        self.rewards = np.load(self.dir_save + "rewards.npy")
        self.terminals = np.load(self.dir_save + "terminals.npy")


#Se define la clase DQNReplayMemoria propia para la implementación de la red DQN.
class DQNReplayMemory(ReplayMemory):
    #Se inicializa la clase padre con el objeto config. 
    #Se inicializandos arreglo pre y post de acuerdo a los tamaños del batch, del historial y dimensiones
    #de las pantallas definidos en el objeto config
    def __init__(self, config):
        super(DQNReplayMemory, self).__init__(config)

        self.pre = np.empty((self.config.batch_size, self.config.history_len, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.post = np.empty((self.config.batch_size, self.config.history_len, self.config.screen_height, self.config.screen_width), dtype=np.uint8)

    #Se encarga de retornar un estado de la memoria de repetición 
    def getState(self, index):

        index = index % self.count  #Se define un índice de acuerdo a atributo count y al argumento
        if index >= self.config.history_len - 1: # si indice es mayor o igual que el valor definidio en config
            a = self.screens[(index - (self.config.history_len - 1)):(index + 1), ...] #Se retorna el intervalo de screen definido
            return a 
        else: #Si e índice es menor se retornar retornan los últimos screens
            indices = [(index - i) % self.count for i in reversed(range(self.config.history_len))]
            return self.screens[indices, ...]

    #Se encarga de agregar un elemento a la memoria de repetición en las posiciones.
    def add(self, screen, reward, action, terminal):
        assert screen.shape == (self.config.screen_height, self.config.screen_width)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current] = screen
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size

    #Se encarga de seleccionar un muestra del tamaño definido en config para el entrenamiento
    def sample_batch(self):
        assert self.count > self.config.history_len

        indices = []
        while len(indices) < self.config.batch_size: #Se itera mientras aun haya que seleccionar elementos

            while True: #Se itera sin fin
                index = random.randint(self.config.history_len, self.count-1) #Selecciona un índice aleatorio

                if index >= self.current and index - self.config.history_len < self.current: #Si este indice no cumple la condción
                    continue #genera otro índice

                if self.terminals[(index - self.config.history_len): index].any(): #Si ninguno de los valores del intervalo en termina es verdadero
                    continue
                break #se deja de iterar
            self.pre[len(indices)] = self.getState(index - 1) #Calcula los screens anteriores xt para el algoritmo de repetición de experiencias
            self.post[len(indices)] = self.getState(index) #Calcula los screens posteriores xt+1
            indices.append(index)

        actions = self.actions[indices]  #Asigna las acciones para el algoritmo
        rewards = self.rewards[indices] #Asigna las recompensas para los estados 
        terminals = self.terminals[indices] #Asigna los valores terminales para el etrenamiento

        return self.pre, actions, rewards, self.post, terminals

class DRQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DRQNReplayMemory, self).__init__(config)

        self.timesteps = np.empty((self.config.mem_size), dtype=np.int32)
        self.states = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.actions_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.rewards_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.terminals_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))

    def add(self, screen, reward, action, terminal, t):
        assert screen.shape == (self.config.screen_height, self.config.screen_width)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current] = screen
        self.timesteps[self.current] = t
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size


    def getState(self, index):
        a = self.screens[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a

    def get_scalars(self, index):
        t = self.terminals[index - (self.config.min_history + self.config.states_to_update + 1): index]
        a = self.actions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        r = self.rewards[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a, t, r

    def sample_batch(self):
        assert self.count > self.config.min_history + self.config.states_to_update

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.min_history, self.count-1)
                if index >= self.current and index - self.config.min_history < self.current:
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    continue
                break
            self.states[len(indices)] = self.getState(index)
            self.actions_out[len(indices)], self.terminals_out[len(indices)], self.rewards_out[len(indices)] = self.get_scalars(index)
            indices.append(index)


        return self.states, self.actions_out, self.rewards_out, self.terminals_out
