import os
import pickle
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from itertools import product
from typing import Optional, Dict
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

@dataclass
class JALAMAgentConfig:
    """
    Configuración para el agente JALAM.

    alpha: tasa de aprendizaje.
    gamma: factor de descuento.
    start_epsilon: tasa inicial de exploración.
    min_epsilon: tasa mínima de exploración.
    episodes: cantidad total de episodios para planificar el decay.
    exploring: indica si explora.
    seed: semilla para reproducibilidad.
    """
    alpha: float = 1.0
    gamma: float = 0.9
    start_epsilon: float = 1.0
    min_epsilon: float = 0.0
    episodes: int = 10000
    exploring: bool = True
    seed: Optional[int] = None

# Funciones para defaultdict sin lambdas

def default_q_value():
    return 0.0

def default_q_table():
    return defaultdict(default_q_value)

def default_count_value():
    return 0

def default_count_table():
    return defaultdict(default_count_value)

class JALAMAgent(Agent):
    """
    Agente JALAM optimizado: usa tuplas de acciones conjuntas y estima políticas de otros agentes.
    """
    def __init__(
        self,
        agent: AgentID,
        action_spaces,
        game: SimultaneousGame,
        config: JALAMAgentConfig,
        initial_Q: Optional[defaultdict] = None,
        initial_counts: Optional[Dict[AgentID, defaultdict]] = None
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
        self.start_epsilon = config.start_epsilon
        self.min_epsilon = config.min_epsilon
        self.episodes = config.episodes
        self.epsilon = config.start_epsilon
        self.epsilon_decay = (self.min_epsilon / self.start_epsilon) ** (1.0 / self.episodes)
        self.exploring = config.exploring

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

    def procesar_observacion(self, obs) -> Optional[tuple]:
        if isinstance(obs, np.ndarray):
            return tuple(obs.tolist())
        if isinstance(obs, list):
            return tuple(obs)
        return obs

    def reset(self, epsilon: Optional[float] = None) -> None:
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = self.start_epsilon
        self.epsilon_decay = (self.min_epsilon / self.start_epsilon) ** (1.0 / self.episodes)
        self.last_state = self.procesar_observacion(self.game.observe(self.agent))
        self.last_action = None

    def _pi_j(self, j, a_j, s):
        """
        Devuelve la probabilidad estimada de que el jugador j juegue a_j en el estado s,
        basada en los counts. Si no hay datos, asume una política uniforme.
        """
        if s not in self.counts[j]:
            # política uniforme si no se ha observado el estado
            return 1.0 / len(self.A[j])
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
            q_value = self.Q.get(s, {}).get(joint, 0.0)  # 🔒 Protección contra KeyError
            v += q_value * p
        return v

    def action(self) -> int:
        """Epsilon-greedy sobre AV_i."""
        state = self.last_state
        if self.exploring and random.random() < self.epsilon:
            a = random.choice(self.A[self.agent])
        else:
            if state not in self.Q:
                # No conocemos Q[state], actuamos al azar
                a = random.choice(self.A[self.agent])
            else:
                best_v = None
                best_actions = []
                for a_i in self.A[self.agent]:
                    v = self.AV_i(state, a_i)
                    if best_v is None or v > best_v:
                        best_v, best_actions = v, [a_i]
                    elif v == best_v:
                        best_actions.append(a_i)
                a = random.choice(best_actions)

        self.last_action = int(a)
        return self.last_action

    def update(self, joint_action) -> None:
        """Actualiza counts y Q-learning."""
        s = self.last_state
        # formar tupla joint
        if isinstance(joint_action, dict):
            joint = tuple(joint_action[j] for j in self.players)
        else:
            joint = joint_action
        r = self.game.reward(self.agent)
        s_next = self.procesar_observacion(self.game.observe(self.agent))

        # actualizar estimación de políticas
        for idx, j in enumerate(self.players):
            if j == self.agent:
                continue
            if s not in self.counts[j]:
                self.counts[j][s] = {}
            if joint[idx] not in self.counts[j][s]:
                self.counts[j][s][joint[idx]] = 0
            self.counts[j][s][joint[idx]] += 1

        # Q-learning
        if s not in self.Q:
            self.Q[s] = {}
        if joint not in self.Q[s]:
            self.Q[s][joint] = 0.0
        curr = self.Q[s][joint]
        best_next = max(self.AV_i(s_next, a_i) for a_i in self.A[self.agent])
        target = r + self.gamma * best_next
        self.Q[s][joint] = curr + self.alpha * (target - curr)

        if self.exploring:
            self.epsilon *= self.epsilon_decay

        self.last_state = s_next

    def save(self, path: str) -> None:
        """Guarda Q y counts como pickle."""
        def to_dict(d):
            if isinstance(d, defaultdict):
                return {k: to_dict(v) for k, v in d.items()}
            return d

        data = {
            'Q': to_dict(self.Q),
            'counts': {j: to_dict(self.counts[j]) for j in self.counts}
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Carga Q y counts desde pickle."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)

            def to_q_default(d):
                q_dd = defaultdict(default_q_table)
                for state, actions in d.items():
                    q_dd[state].update(actions)
                return q_dd

            def to_count_default(d):
                c_dd = defaultdict(default_count_table)
                for state, cnts in d.items():
                    c_dd[state].update(cnts)
                return c_dd

            self.Q = to_q_default(data['Q'])
            self.counts = {j: to_count_default(data['counts'][j]) for j in data['counts']}
