from src.dqn_agent import DQNAgent
from src.config import  GymConfig
import sys

import argparse


#Se encarga de crear el agente que se entrena
class Main():

    def __init__(self, net_type, conf):
  
        self.agent = DQNAgent(conf)

    #Se encarga de realizar todo el proceso de entrenamiento del agente
    def train(self, steps):
        self.agent.train(steps)

    #Se encarga de jugar un episodio con lo aprendido por la red neuronal especificiada.
    def play(self, steps, net_path):
        self.agent.play(steps, net_path)

#Recibe los parámetros cuando se ejecuta el programa por consola.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRQN")
    parser.add_argument("--gym", type=str, default="gym", help="Type of the environment. Can either be 'gym' or 'retro'")
    parser.add_argument("--network_type", type=str, default="dqn", help="Type of the network to build, can either be 'dqn' or 'drqn'")
    parser.add_argument("--env_name", type=str, default="Breakout-v0", help="Name of the gym/retro environment used to train the agent")
    parser.add_argument("--retro_state", type=str, default="Start", help="Name of the state (level) to start training. This is only necessary for retro envs")
    parser.add_argument("--train", type=str, default="True", help="Whether to train a network or to play with a given network")
    parser.add_argument("--model_dir", type=str, default="saved_session/net/", help="directory to save the model and replay memory during training")
    parser.add_argument("--net_path", type=str, default="", help="path to checkpoint of model")
    parser.add_argument("--steps", type=int, default=50000000, help="number of frames to train")
    args, remaining = parser.parse_known_args()

    
    conf = GymConfig()                      #Se crea la configuración
    conf.env_name = args.env_name           #Se nombre del ambiente en el objeto conf
    conf.network_type = args.network_type   #Se define el tipo de red neuronal en el objeto conf
    conf.train = args.train                 #Se define que se hará el proceso de entrenamiento de la red
    conf.dir_save = args.model_dir          #Se define la ruta donde se almacena el modelo entrenado en el objeto conf
    conf.train_steps = args.steps           #Se define la cantidad de timesteps que dura el entrenamiento en el objeto conf
    main = Main(conf.network_type, conf)    #Se crean un objeto Main, con los atríbutos del objeto conf.

    if conf.train == "True":                #Si se está entrenando la red
        print(conf.train)
        main.train(conf.train_steps)        #El objeto main ejecuta train por la cantidad de pasos de tiempo definida.
    else:                                   #Si no, el objeto main juego por 100000 time steps de acuerdo al modelo entrenado en la ruta dada.
        assert args.net_path != "", "Please specify a net_path using the option --net_path"
        main.play(100, args.net_path)



