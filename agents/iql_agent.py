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
    lr: tasa de aprendizaje (alpha).
    gamma: factor de descuento.
    exploring: indica si explora o no.
    seed: semilla aleatoria para reproducibilidad.
    """
    start_epsilon: float = 1.0
    min_epsilon: float = 0.01
    episodes: int = 10000
    lr: float = 1.0
    gamma: float = 0.9
    exploring: bool = True
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
        initial: Optional[Dict[tuple, Dict[int, float]]] = None
    ) -> None:
        super().__init__(game=game, agent=agent)
        # reproducibilidad para numpy y random
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        # tabla Q como dict de dict
        self.Q: Dict[tuple, Dict[int, float]] = {} if initial is None else initial
        self.last_state: Optional[tuple] = None
        self.last_action: Optional[int] = None

        # hiperparámetros
        self.lr = config.lr
        self.gamma = config.gamma
        self.exploring = config.exploring
        # exploración: start, mínimo y decay calculado
        self.epsilon = config.start_epsilon
        self.min_epsilon = config.min_epsilon
        self.episodes = config.episodes
        # decay por episodio
        self.epsilon_decay = (config.min_epsilon / config.start_epsilon) ** (1.0 / config.episodes)

        # acciones posibles según el espacio propio del agente
        idx = self.game.agents.index(self.agent)
        self.possible_actions = list(range(self.game.env.action_space[idx].n))

    def reset(self, epsilon: float = None) -> None:
        """Reinicia el agente y opcionalmente ajusta el epsilon inicial."""
        if epsilon is not None:
            self.epsilon = epsilon if self.exploring else 0.0
            self.epsilon_decay = (self.min_epsilon / self.start_epsilon) ** (1.0 / self.episodes)
        self.last_state = self.procesar_observacion(self.game.observe(self.agent))

    def argmax_or_random(self, d: Dict[int, float]) -> int:
        if d:
            return max(d, key=d.get)
        return random.choice(self.possible_actions)

    def bestresponse(self, state: tuple) -> int:
        """Epsilon-greedy usando Q almacenada."""
        if state is None or (self.exploring and random.random() < self.epsilon):
            return random.choice(self.possible_actions)
        return self.argmax_or_random(self.Q.get(state, {}))

    def update(self) -> None:

        if not(self.exploring):
            return

        new_state = self.procesar_observacion(self.game.observe(self.agent))
        reward = self.game.reward(self.agent)

        if self.last_state is not None and new_state is not None:
            # valor futuro máximo
            next_actions = self.Q.get(new_state, {})
            V_sp = max(next_actions.values()) if next_actions else 0.0

            # preparar acciones del estado anterior
            state_actions = self.Q.setdefault(self.last_state, {})
            old_q = state_actions.get(self.last_action, 0.0)

            # update de Q-learning
            state_actions[self.last_action] = old_q + self.lr * (
                reward + self.gamma * V_sp - old_q
            )

        # decaimiento de epsilon por episodio
        if self.exploring:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # actualizar estado previo
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
        """Guarda la tabla Q en disco como pickle de un dict."""
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load_q(self, path: str) -> None:
        """Carga tabla Q desde pickle, si existe."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.Q = pickle.load(f)
