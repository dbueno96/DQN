import gym
from src.utils import resize, rgb2gray
import numpy as np


#Se define la clase GymWrapper que se encarga de todos los 
#aspectos propios del ambiente de juego
class GymWrapper():

    #Se inicializa en el ambiente gym, dimensiones de pantalla, 
    #una recomentan un boolean, las vidas, el frameskip, un comienzo aleatorio, 
    #y el espacio de accioones legales.
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.screen_width, self.screen_height = config.screen_width, config.screen_height
        self.reward = 0
        self.terminal = True
        self.info = {'ale.lives': 0}
        self.env.env.frameskip = config.frame_skip
        self.random_start = config.random_start
        self.action_space = self.env.action_space
        self._screen = np.empty((210, 160), dtype=np.uint8)

    #Se encarga de iniciar un episodio nuevo cuando el agente
    #pierde todas las vidas.
    def new_game(self):
        if self.lives == 0:
            self.env.reset()
        self._step(0)
        self.reward = 0
        self.action = 0

    #Se encarga de jugar un episodio por una cantidad aleatoria de timesteps
    def new_random_game(self):
        self.new_game()
        for _ in range(np.random.randint(0, self.random_start)):
            self._step(0)

    #Se encarga de ejecutar la acción del argumento, y almacenar la información recibida del ambiente
    def _step(self, action):
        self.action = action
        _, self.reward, self.terminal, self.info = self.env.step(action)

    #Se encarga de tomar una acción aleatoria del conjunto de acciones legale.
    def random_step(self):
        return self.action_space.sample()

    #Se encarga de llamar al método que ejecuta la acción,
    #y verificar si se pierde una vida en esa transición
    def act(self, action):
        lives_before = self.lives
        self._step(action)
        if self.lives < lives_before:
            self.terminal = True

    #Se encarga de llamar al método que ejecuta la acción,
    #y verificar si se pierde una vida en esa transición y renderizar.
    def act_play(self, action):
        lives_before = self.lives
        self._step(action)
        self.env.render()
        if self.lives < lives_before:
            self.terminal = True
    
    #Se encargade iniciar un nuevo episodio de juego.
    def new_play_game(self):
        self.new_game()
        self._step(1)


    #Define la propieda screen tomada de ALE, sobre lo que está implementado openAI gym
    @property
    def screen(self):
        self._screen = self.env.env.ale.getScreenGrayscale(self._screen)
        a = resize(self._screen ,(self.screen_height, self.screen_width))
        return a
    
    #Define a propiedad de las vidas.
    @property
    def lives(self):
        return self.info['ale.lives']

