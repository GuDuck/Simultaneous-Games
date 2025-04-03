from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        self.count: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random 
        #
        if initial is not None:
            for agent in game.agents:
                self.count[agent] = initial[agent]
        else:
            for agent in game.agents:
                self.count[agent] = np.random.rand(game.num_actions(agent))
                
        

        self.learned_policy: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar learned_policy usando de count
        # 
        for agent in game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])
     

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta 
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #
        for actions in product(*agents_actions):   
            g.reset()
            g.step(dict(zip(g.agents, actions)))
            rewards[actions] = g.reward(self.agent)

        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente. 
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        for actions, reward in rewards.items():
            prob = 1
            for agent in self.game.agents:
                if agent != self.agent:
                    prob *= self.learned_policy[agent][actions[int(agent[-1])]]
            action = actions[int(self.agent[-1])]
            utility[action] += reward * prob
    
        return utility
    
    def bestresponse(self):
        a = None
        #
        # TODO: retornar la acción de mayor utilidad
        #
        utility = self.get_utility()
        a = np.argmax(utility)
 
        return a
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    