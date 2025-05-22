
import numpy as np
import random
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Dict
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

@dataclass
class IQLAgentConfig:
    """
    Configuración para el agente IQL.

    start_epsilon: tasa inicial de exploración.
    min_epsilon: tasa mínima de exploración.
    episodes: cantidad total de episodios para planificar el decay.
    alpha: tasa de aprendizaje.
    gamma: factor de descuento.
    learn: indica si explora o no.
    seed: semilla aleatoria para reproducibilidad.
    """
    start_epsilon: float = 1.0
    min_epsilon: float = 0.01
    epsilon_decay: float = 1.0
    alpha: float = 1.0
    gamma: float = 0.9
    learn: bool = True
    seed: Optional[int] = None

class IQLAgent(Agent):
    """
    Agente basado en Independent Q-Learning (IQL) para entornos multijugador.
    Calcula internamente el epsilon_decay en base a la configuración.
    """

    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        config: IQLAgentConfig,
        initial: Optional[Dict[tuple, ndarray]] = None
    ) -> None:
        super().__init__(game=game, agent=agent)

        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        self.Q: Dict[tuple, ndarray] = {} if initial is None else initial
        self.last_state: Optional[tuple] = None
        self.last_action: Optional[int] = None

        self.alpha = config.alpha
        self.gamma = config.gamma
        self.learn = config.learn

        self.epsilon = config.start_epsilon
        self.min_epsilon = config.min_epsilon
        self.epsilon_decay = config.epsilon_decay

        idx = self.game.agents.index(self.agent)
        self.possible_actions = list(range(self.game.env.action_space[idx].n))

    def reset(self) -> None:
        self.last_state = self.procesar_observacion(self.game.observe(self.agent))

    def argmax_or_random(self, state) -> int:
        if state not in self.Q:
            na = len(self.possible_actions)
            self.Q[state] = np.zeros(na) # np.ones(na) * (1/na)

        q_values = self.Q[state]
        best_choices = np.argwhere(q_values == np.max(q_values)).flatten()

        return int(np.random.choice(best_choices))

    def epsilon_greedy(self, state: tuple) -> int:
        if self.learn and random.random() < self.epsilon:
            return random.choice(self.possible_actions)
        return self.argmax_or_random(state)

    def update(self) -> None:
        if not self.learn:
            return

        new_state = self.procesar_observacion(self.game.observe(self.agent))
        reward = self.game.reward(self.agent)

        if new_state is not None: # self.last_state is not None and 
            if new_state not in self.Q:
                self.Q[new_state] = np.zeros(len(self.possible_actions))
            if self.last_state not in self.Q:
                self.Q[self.last_state] = np.zeros(len(self.possible_actions))

            V_sp = np.max(self.Q[new_state])

            self.Q[self.last_state][self.last_action] += self.alpha * (
                reward + self.gamma * V_sp - self.Q[self.last_state][self.last_action]
            )

    def action(self) -> int:

        self.last_state = self.procesar_observacion(self.game.observe(self.agent))
        self.last_action = self.epsilon_greedy(self.last_state)

        if self.learn:
            self.epsilon = max(self.min_epsilon, self.epsilon * (self.epsilon_decay))

        return self.last_action

    def procesar_observacion(self, observation: Optional[ndarray]) -> Optional[tuple]:
        if observation is None:
            return None
        return tuple(observation.astype(np.int32))

    def save_q(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load_q(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.Q = pickle.load(f)
