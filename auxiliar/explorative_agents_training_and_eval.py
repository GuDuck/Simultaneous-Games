import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from base.game import SimultaneousGame
from base.agent import Agent
from games.foraging import Foraging
from agents.jalam_agent import JALAMAgent


def play_episode(game: SimultaneousGame,
                 agents: Dict[str, type[Agent]],
                 verbose: Optional[bool]=False,
                 render: Optional[bool]=False,
                 agent_aliases: Optional[Dict[str, str]]=None
                 ) -> Dict[str, float]:

    # Initialize the game
    game.reset()
    step_count = 0

    # Initialize each agent
    for agent in game.agents:
        agents[agent].reset()

    # Print initial observations if verbose is enabled
    if verbose:
        print(f"Step: {step_count}")
        for agent in game.agents:
            agent_name = agent if agent_aliases is None else agent_aliases[agent]
            print(f"Agent {agent_name} observe: {game.observe(agent)}")

    # Initialize rewards for each agent
    cum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))

    # render the game if required
    if render:
        game.render()
        time.sleep(0.5)

    while not game.done():

        step_count += 1
        
        # Get actions from each agent
        actions = {}
        for agent in game.agents:
            actions[agent] = agents[agent].action()
             
        # Perform the actions in the game
        game.step(actions)

        # Update the cum_rewards for each agent
        for agent in game.agents:
            cum_rewards[agent] += game.reward(agent)

        # Print actions, rewards and next state if verbose is enabled
        if verbose:
            print(f"Step: {step_count}")
            for agent in game.agents:
                agent_name = agent if agent_aliases is None else agent_aliases[agent]
                print(f"Agent {agent_name} action: {actions[agent]} - {game.action_set[actions[agent]]}")
                print(f"Agent {agent_name} reward: {game.reward(agent)}")
                print(f"Agent {agent_name} observe: {game.observe(agent)}")
            
        if render:
            game.render()
            time.sleep(0.5)
    
        for agent in game.agents:
            # Update the agent with the last observation
            if isinstance(agents[agent], JALAMAgent):
                agents[agent].update(actions)
            else:
                agents[agent].update()
    
    return cum_rewards

def run(game: SimultaneousGame,
        agents: Dict[str, type[Agent]],
        episodes: Optional[int]=1,
        verbose: Optional[bool]=False,
        render: Optional[bool]=False,
        agent_aliases: Optional[Dict[str, str]]=None
        ) -> Dict[str, float]:
    
    # Contenedor de las recompensas acumuladas de todos los episodios
    sum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))

    # Iterar sobre la cantidad de episodios
    for _ in range(episodes):
        # Simular y hallar recompensas acumuladas del episodio
        cum_rewards = play_episode(game, agents, verbose, render, agent_aliases)  
        
        # Acumular recompensas de los episodios
        for agent in game.agents:
            sum_rewards[agent] += cum_rewards[agent]

    # Imprimir información
    if verbose:
        print(f"Average rewards over {episodes} episodes:")
        for agent in game.agents:
            agent_name = agent if agent_aliases is None else agent_aliases[agent]
            print(f"Agent {agent_name}: {sum_rewards[agent] / episodes}")

    # Retorna la suma acumulada de recompensas de todos los episodios
    return sum_rewards

def train(game: SimultaneousGame,
          agents: Dict[str, type[Agent]],
           train_config: Dict[str, int],
            progress: Optional[bool]=False,
            verbose: Optional[bool]=False,
            render: Optional[bool]=False,
            agent_aliases: Optional[Dict[str, str]]=None
            ) -> Dict[str, List[float]]:
    
    # Tomo configuración del entrenamiento
    iterations = train_config["iterations"]
    episodes = train_config["episodes"]

    # Contenedor de las recompensas acumuladas promedio
    average_rewards = dict(map(lambda agent: (agent, []), game.agents))

    # Loop en iteraciones
    for i in range(1, iterations+1):
        
        # Correr el nro de episodios y obtener recompensas acumuladas de los agentes
        sum_rewards = run(game, agents, episodes, verbose, render, agent_aliases)

        # Agregar a lista y promediar sobre la cantidad de episodios
        for agent in game.agents:
            average_rewards[agent].append(sum_rewards[agent] / episodes)

        # Imprimir progreso
        if progress and (i % 10 == 0):
            for agent in game.agents:
                agent_name = agent if agent_aliases is None else agent_aliases[agent]
                print(f"Agent {agent_name}: {average_rewards[agent][-1]}")
    
    # Imprimir progreso
    if progress:
        print(f"Last average rewards over {iterations} iterations ({iterations * episodes} episodes):")
        for agent in game.agents:
            agent_name = agent if agent_aliases is None else agent_aliases[agent]
            print(f"Agent {agent_name}: {average_rewards[agent][-1]}")

    # Retorna las recompensas acumuladas medias de cada iteración
    return average_rewards

def entrenar_agente_tarea(args: tuple, seed: Optional[int]=None):

    # Recibe la configuración y otros parámetros a través de args (tupla)
    config_id, config, agent_classes, agent_aliases = args

    # Genera una instancia del juego
    game = Foraging(config=config['game'], seed=seed)

    # Crear a los agentes
    agents = {
        agent: agent_classes[agent](game=game, agent=agent, config=config[agent], action_spaces=game.action_spaces) 
        if issubclass(agent_classes[agent], JALAMAgent) else 
        agent_classes[agent](game=game, agent=agent, config=config[agent]) for agent in game.agents}

    # Resetear juego e imprimir observaciones de los agentes
    game.reset()
    for agent in game.agents:
        agent_name = agent if agent_aliases is None else agent_aliases[agent]
        print(f"Agent: {agent_name}")
        print(f"Observation: {game.observe(agent)}")

    # Entrenar y obtener las recompensas acumuladas medias por episodio de cada iteración
    average_rewards = train(game, agents, train_config=config['train_config'], progress=True, verbose=False, render=False, agent_aliases=agent_aliases)

    # Generar diccionario con los resultados e información del entrenamiento (configuración, etc.)
    resultados_dict = {
        f"experimento_{config_id}": {
            'agents': agents,
            'game': game,
            'config': config,
            'average_rewards': average_rewards,
            'agent_aliases': agent_aliases
        }
    }

    # Guardar resultados
    with open(f"experiments/results_{config_id}.pkl", "wb") as f:
        pickle.dump(resultados_dict, f)

    # Devolver resultados
    return resultados_dict

def load_experiments(experiments_names: List[str]) -> Dict[str, Dict]:

    experiments = {}
    for experiment_name in experiments_names:
        print(f'Intentando cargar el experimento: {experiment_name}')
        path = f"./experiments/results_{experiment_name}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                experiment = pickle.load(f)
                for key, val in experiment.items():
                    experiments[key] = val
            print(f'{experiment_name} cargado con éxito.')
        else:
            Warning(f"No existe el archivo: {path}. Returning None.")
            return None

    return experiments

def eval_experiments(experiments_names: List[str]) -> Dict[str, Dict]:

    experiments = load_experiments(experiments_names)

    for experiment_name, experiment in experiments.items():
        print(f"Experiment config: {experiment['config']['game']}")

        average_rewards = experiment['average_rewards']
        experiment_agents = experiment['agents']
        agent_aliases = None if 'agent_aliases' not in experiment else experiment['agent_aliases']

        for agent, rewards in average_rewards.items():
            agent_name = agent if agent_aliases is None else agent_aliases[agent]
            plt.plot(rewards, label=agent_name)

        plt.xlabel('Iterations')
        plt.ylabel('Rewards')
        plt.title(f'Experiment {experiment_name} - Rewards per Agent')
        plt.legend()
        plt.show()

        for agent, rewards in average_rewards.items():
            agent_name = agent if agent_aliases is None else agent_aliases[agent]
            plt.plot(np.cumsum(rewards), label=agent_name)

        plt.xlabel('Iterations')
        plt.ylabel('Average Cumulative Rewards')
        plt.title(f'Experiment {experiment_name} - Average Cumulative Rewards per Agent')
        plt.legend()
        plt.show()

        for agent in experiment_agents.keys():
            experiment_agents[agent].learn = False

        game = experiment['game'].clone()
        game.reset()

        play_episode(game, experiment_agents, verbose=True, render=True)
        game.close()

    return experiments
