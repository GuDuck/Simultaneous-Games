import numpy as np
import random
import os
import pickle
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

@dataclass
class IQLAgentConfig:
    """
    Configuración para el agente IQL.

    lr: tasa de aprendizaje (alpha).
    gamma: factor de descuento.
    epsilon: tasa de exploración inicial.
    epsilon_decay: factor de decaimiento de epsilon.
    exploring: indica si explora o no.
    seed: semilla aleatoria para reproducibilidad.
    """
    lr: float = 1.0
    gamma: float = 0.9
    epsilon: float = 0.5
    epsilon_decay: float = 1.0
    exploring: bool = True
    seed: Optional[int] = None

def default_q_value():
    return 0.0

def default_q_table():
    return defaultdict(default_q_value)

class IQLAgent(Agent):
    """
    Agente basado en Independent Q-Learning (IQL) para entornos multijugador.
    """

    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        config: IQLAgentConfig,
        initial: Optional[defaultdict] = None
    ) -> None:
        """
        Inicializa el agente IQL usando un objeto de configuración.
        """
        super().__init__(game=game, agent=agent)
        # semilla para reproducibilidad
        if config.seed is not None:
            np.random.seed(seed=config.seed)

        # tabla Q
        self.Q = defaultdict(default_q_table) if initial is None else initial
        self.last_state: Optional[tuple] = None
        self.last_action: Optional[int] = None

        # hiperparámetros
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.exploring = config.exploring

        # acciones posibles según el entorno
        self.possible_actions = list(range(self.game.env.action_space[0].n))

    def reset(self, epsilon: float = None) -> None:
        """
        Reinicia el agente y, opcionalmente, ajusta epsilon.
        """
        if epsilon is not None:
            self.epsilon = epsilon
        self.last_state = self.procesar_observacion(self.game.observe(self.agent))

    def argmax_or_random(self, d: dict[int, float]) -> int:
        if d:
            return max(d, key=d.get)
        return random.choice(self.possible_actions)

    def bestresponse(self, state: tuple) -> int:
        if (self.last_state is None) or (self.exploring and random.random() < self.epsilon):
            return random.choice(self.possible_actions)
        return self.argmax_or_random(self.Q[state])

    def update(self) -> None:
        new_state = self.procesar_observacion(self.game.observe(self.agent))
        reward = self.game.reward(self.agent)

        if self.last_state is not None and new_state is not None:
            Q_sp = self.Q[new_state]
            V_sp = max(Q_sp.values()) if Q_sp else 0.0
            self.Q[self.last_state][self.last_action] += self.lr * (
                reward + self.gamma * V_sp - self.Q[self.last_state][self.last_action]
            )

        if self.exploring:
            self.epsilon *= self.epsilon_decay

        self.last_state = new_state

    def action(self) -> int:
        self.last_action = self.bestresponse(self.last_state)
        return self.last_action

    def procesar_observacion(self, observation: Optional[ndarray]) -> Optional[tuple]:
        if observation is None:
            return None
        mask = (np.arange(1, len(observation) + 1) % 3) != 0
        return tuple(observation[mask])

    def save_q(self, path: str) -> None:
        def to_dict(d):
            if isinstance(d, defaultdict):
                return {k: to_dict(v) for k, v in d.items()}
            return d

        with open(path, "wb") as f:
            pickle.dump(to_dict(self.Q), f)

    def load_q(self, path: str) -> None:
        def to_defaultdict(d: dict) -> defaultdict:
            q_dd = defaultdict(lambda: defaultdict(lambda: 0.0))
            for state, actions in d.items():
                q_dd[state].update(actions)
            return q_dd

        if os.path.exists(path):
            with open(path, "rb") as f:
                q_plain = pickle.load(f)
            self.Q = to_defaultdict(q_plain)
