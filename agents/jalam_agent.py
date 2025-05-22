import os
import pickle
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from itertools import product
from typing import Optional, Dict
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

@dataclass
class JALAMAgentConfig:
    """
    Configuración para el agente JAL-AM.

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

class JALAMAgent(Agent):
    def __init__(
        self,
        agent: AgentID,
        action_spaces,
        game: SimultaneousGame,
        config: JALAMAgentConfig,
        initial_Q: Optional[Dict] = None,
        initial_counts: Optional[Dict[AgentID, Dict]] = None
    ) -> None:
        super().__init__(game=game, agent=agent)

        # reproducibilidad
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        # referencias
        self.game = game
        self.agent = agent
        self.players = list(game.agents)
        self.idx = self.players.index(self.agent)

        # hiperparámetros
        self.alpha = config.alpha
        self.gamma = config.gamma

        self.epsilon = config.start_epsilon
        self.min_epsilon = config.min_epsilon
        self.epsilon_decay = config.epsilon_decay
        
        self.learn = config.learn

        # espacios de acción
        self.A = { j: np.arange(action_spaces[j].n, dtype=np.int8) for j in self.players }
        joint_tuples = list(product(*(self.A[j] for j in self.players)))
        self.joint_actions = {
            a_i: [jt for jt in joint_tuples if jt[self.idx] == a_i]
            for a_i in self.A[self.agent]
        }

        # tablas Q y counts
        self.Q: Dict[tuple, Dict[tuple, float]] = {} if initial_Q is None else initial_Q
        if initial_counts is None:
            self.counts: Dict[AgentID, Dict[tuple, Dict[int, int]]] = { j: {} for j in self.players if j != self.agent }
        else:
            self.counts = initial_counts

        # estado y acción previos
        self.last_state: Optional[tuple] = None
        self.last_action: Optional[int] = None

    def procesar_observacion(self, observation: Optional[ndarray]) -> Optional[tuple]:
        if observation is None:
            return None
        return tuple(observation.astype(np.int32))

    def reset(self) -> None:
        self.last_state = self.procesar_observacion(self.game.observe(self.agent))

    def _pi_j(self, j, a_j, s):
        # política uniforme si no se ha observado el estado
        if s not in self.counts[j]:
            return 1.0 / len(self.A[j])
        
        # Estima política
        cnts = self.counts[j][s]
        total = sum(cnts.values())

        if total == 0:
            return 1.0 / len(self.A[j])
        
        return cnts.get(a_j, 0) / total

    def AV_i(self, s, a_i) -> float:
        v = 0.0
        for joint in self.joint_actions[a_i]:
            p = 1.0
            for idx, j in enumerate(self.players):
                if j == self.agent:
                    continue
                p *= self._pi_j(j, joint[idx], s)
            
            if s not in self.Q:
                self.Q[s] = {joint: 0.0 for joints in self.joint_actions.values() for joint in joints}

            q_value = self.Q[s][joint]
            v += q_value * p
        return v

    def action(self) -> int:
        self.last_state = self.procesar_observacion(self.game.observe(self.agent))
        if self.learn and random.random() < self.epsilon:
            a = random.choice(self.A[self.agent])
        else:
            if self.last_state not in self.Q:
                a = random.choice(self.A[self.agent])
            else:
                best_v = None
                best_actions = []
                for a_i in self.A[self.agent]:
                    v = self.AV_i(self.last_state, a_i)
                    if best_v is None or v > best_v:
                        best_v, best_actions = v, [a_i]
                    elif v == best_v:
                        best_actions.append(a_i)

                a = random.choice(best_actions)

        self.last_action = int(a)

        if self.learn:
            self.epsilon = max(self.min_epsilon, self.epsilon * (self.epsilon_decay))

        return self.last_action

    def update(self, joint_action) -> None:

        if not self.learn:
            return

        # Observar
        s = self.last_state
        r = self.game.reward(self.agent)
        s_next = self.procesar_observacion(self.game.observe(self.agent))

        # formar tupla joint
        if isinstance(joint_action, dict):
            joint = tuple(joint_action[j] for j in self.players)
        else:
            joint = joint_action

        # actualizar estimación de políticas
        for idx, j in enumerate(self.players):
            if j == self.agent:
                continue
            if s not in self.counts[j]:
                self.counts[j][s] = {}
            if joint[idx] not in self.counts[j][s]:
                self.counts[j][s][joint[idx]] = 0
            self.counts[j][s][joint[idx]] += 1

        # Joint Q-learning
        if s not in self.Q:
            self.Q[s] = {joint: 0.0 for joints in self.joint_actions.values() for joint in joints}
        if joint not in self.Q[s]:
            self.Q[s][joint] = 0.0
        best_next = max(self.AV_i(s_next, a_i) for a_i in self.A[self.agent])
        self.Q[s][joint] += self.alpha * (r + self.gamma * best_next - self.Q[s][joint])

    def save(self, path: str) -> None:
        data = {
            'Q': self.Q,
            'counts': self.counts
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.Q = data['Q']
            self.counts = data['counts']
