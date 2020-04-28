from src.agent import BaseAgent
from src.history import History
from src.replay_memory import DQNReplayMemory
from src.networks.dqn import DQN
import numpy as np
from tqdm import tqdm

#Se define la clase DQNAgent que reprenta un agente que aprende por medio
#de una red neuronal DQN
class DQNAgent(BaseAgent):
    #Se inicializa la clase padre con el objeto config, un History,
    #una memoria de repetición, una red neuronal y su arquitectura.
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.history = History(config)
        self.replay_memory = DQNReplayMemory(config)
        self.net = DQN(self.env_wrapper.action_space.n, config)
        self.net.build()
        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

    #Se encarga de procesar una observación del agente.
    def observe(self):
        reward = max(self.min_reward, min(self.max_reward, self.env_wrapper.reward)) #Se encarga de que puntaje en el estado esté entre 1 y -1
        screen = self.env_wrapper.screen #Se toma la screen actual del wrapper
        self.history.add(screen) #Se agrega la screen a la historial
        self.replay_memory.add(screen, reward, self.env_wrapper.action, self.env_wrapper.terminal) #Se agrega los valores a la memoria de repetición
        if self.i < self.config.epsilon_decay_episodes: #Se decrementa el valor de epsilon de acuerdo a la cantidad de timesteps
            self.epsilon -= self.config.epsilon_decay
        if self.i % self.config.train_freq == 0 and self.i > self.config.train_start: #se actualizan los pesos ed la red neuronal de acuerdo a los valores en el objeto config.
            state, action, reward, state_, terminal = self.replay_memory.sample_batch() #Se toma una muestra de la memoria de repetición
            q, loss= self.net.train_on_batch_target(state, action, reward, state_, terminal, self.i) #Se actualiza la red neuronal con los valores de la muestra
            self.total_q += q #se incrementa al valor de q
            self.total_loss += loss #se incrementa el valor de pérdida
            self.update_count += 1 #Se contabiliza el entrenamiento 
        if self.i % self.config.update_freq == 0: #Se verifica si es necesario actualiza la red neuronal objetivo 
            self.net.update_target() #Se actualiza la red neurona objetivo

    #Se encarga de definir la política bajo la cual el agente ejecuta un acción.
    def policy(self):
        if np.random.rand() < self.epsilon: #Si epsilon es mayo a un aleatorio
            return self.env_wrapper.random_step() #Se ejecuta una acción aleatoria
        else: #Si no
            state = self.history.get()/255.0 
            a = self.net.q_action.eval({
                self.net.state : [state]
            }, session=self.net.sess)
            return a[0] #Calcula la acción que ofrece mejor puntaje a futuro de acuerdo a la red neuronal

    #Se encarga de realizan el proceso de entrenamiento del agente. 
    def train(self, steps):
        render = False #Se setea la variable para renderiza en false
        self.env_wrapper.new_random_game() #Se inicia un juego y se lleva a un estado aleatorio
        num_game, self.update_count, ep_reward = 0,0,0. #Se inicializan las variables en 0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0. #Se inicializan las varaibles en o
        ep_rewards, actions = [], [] #Se inicializan arreglos vacpios
        t = 0

        for _ in range(self.config.history_len): #Se itera de acuerdo a la longitud de historial
            self.history.add(self.env_wrapper.screen) #Se agregan la primeras iguales screens al historial
        for self.i in tqdm(range(self.i, steps)): #Se itera de acuerdo a los timesteps de entrenamiento
            action = self.policy() #Se selecciona la acción del agente de acuerdo a la poĺítica 
            self.env_wrapper.act(action) #Se eejecuta la acción en el ambiente
            self.observe() #Se procesa la observación.
            if self.env_wrapper.terminal: #Si se llega a un estado terminal 
                t = 0 #reinicia el conteo de timesteps del episodio
                self.env_wrapper.new_random_game() #inicializa un juego y lo lleva un estado aleatorio
                num_game += 1 #Se incrementa la cantidad de episodios jugados
                ep_rewards.append(ep_reward) #Se almacena la recompensa del episodio
                ep_reward = 0. #Se reinicia la recompensa para el sigueinte episodio
            else: #si no es un estado terminal.
                ep_reward += self.env_wrapper.reward #Incrementa la recompensa en el episodio.
                t += 1 #contabiliza el timestep transcurrido
            actions.append(action) #almacena la acción ejecutada 
            total_reward += self.env_wrapper.reward #Incrementa la recomentan total.

            
            if self.i >= self.config.train_start:  #si ya han trasncurridos más timesteps que los especificados en el objeto config
                if self.i % self.config.test_step == self.config.test_step -1: #Si se cumple la condición de acuerdo a los teststeps
                    avg_reward = total_reward / self.config.test_step #Sedeine el promedio de las recompensas sobre los lostest steps
                    avg_loss = self.total_loss / self.update_count #Se define el promedio de pérdida de acuerdo los updates en la red 
                    avg_q = self.total_q / self.update_count #Se define el valor q promedio de acuerdo a los updates en la red

                    try:
                        max_ep_reward = np.max(ep_rewards) #Se almacena la recompensa máxima de los episodios
                        min_ep_reward = np.min(ep_rewards) #Se almacena la recompensa minima de los episodios
                        avg_ep_reward = np.mean(ep_rewards)#Se almaena la recompensa promedio de los episodios
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0 #Si hay un errores se setean a 0

                    sum_dict = { #crea un diccionari con la información definida.
                        'average_reward': avg_reward,
                        'average_loss': avg_loss,
                        'average_q': avg_q,
                        'ep_max_reward': max_ep_reward,
                        'ep_min_reward': min_ep_reward,
                        'ep_num_game': num_game,
                        'learning_rate': self.net.learning_rate,
                        'ep_rewards': ep_rewards,
                        'ep_actions': actions
                    }
                    self.net.inject_summary(sum_dict, self.i) #injecta el diccionario a la red para crear un objeto tf.Summary
                    num_game = 0 #Reinicia los valores guardados en el diccionario
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

            if self.i % 500000 == 0 and self.i > 0: #Cada 500000 timesteps guarda un checkpoint del proceso de entrenamiento de la red
                j = 0
                self.save()
            if self.i % 100000 == 0: #Cada 10000 timesteps renderiza 1000 steps del entrenamiento
                j = 0
                render = True

            # if render:
            #     self.env_wrapper.env.render()
            #     j += 1
            #     if j == 1000:
            #         render = False


    #Se encarga de jugar episodios de acuerdo a la red neuronal entrenada
    def play(self, episodes, net_path):
        self.net.restore_session(path=net_path) #carga los pesos de la red neuronal
        self.env_wrapper.new_game() #Inicia un juego en un estado aleatorio
        i = 0
        for _ in range(self.config.history_len): #sse itera de acuerdo al tamaño del historial
            self.history.add(self.env_wrapper.screen)  #Se almacena las pprimera historias en el historial
        episode_steps = 0
        while i < episodes: 
            a = self.net.q_action.eval({
                self.net.state : [self.history.get()/255.0]
            }, session=self.net.sess) #Calcula la acción que ofrece mejor retorno de acuerdo a la red neuronal
            action = a[0] #Se toma la primera posición, que ofrece el mejor retorno
            self.env_wrapper.act_play(action) #Ejecuta la accion en el ambiente y verifica si perdió vida
            self.history.add(self.env_wrapper.screen) #Almacena el nuevo screen
            episode_steps += 1
            if episode_steps > self.config.max_steps: #Si se alcana la cantidad máxima de steps por episodio
                self.env_wrapper.terminal = True #Marca el episodio como terminado
            if self.env_wrapper.terminal: #Si el episodio está terminado
                episode_steps = 0  #Reiniicia los steps para el nuevo episodio
                i += 1 #contabiliza el episodio jugado
                self.env_wrapper.new_play_game() #Crear un nuevo ambiente en un estado aleatorio
                for _ in range(self.config.history_len): #Se itera sobre el tamaño del historial
                    screen = self.env_wrapper.screen
                    self.history.add(screen) #Almacena la screen en el historia
