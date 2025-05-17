import random
import numpy as np
from collections import defaultdict
from itertools import product

class JALAMAgent:
    def __init__(self, agent_id, action_spaces, game,
                 alpha: float = 1.0, gamma: float = 0.9, exploring=True,
                 epsilon: float = 1.0, epsilon_decay: float = 1.0):
        """
        Agente JALAM optimizado: usa tuplas de acciones conjuntas y preagrupa por acción propia.
        """
        # Referencias
        self.game      = game
        self.agent     = agent_id
        self.players   = list(game.agents)
        # Índice de este agente en la tupla de joint actions
        self.idx       = self.players.index(self.agent)

        # Parámetros de aprendizaje
        self.alpha          = alpha
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_decay  = epsilon_decay
        self.exploring      = exploring

        # Estado y acción previos
        self.last_state     = None
        self.last_action    = None

        # Espacios de acción por agente
        self.A = {
            j: np.arange(action_spaces[j].n, dtype=np.int8)
            for j in self.players
        }

        # Todas las tuplas de acciones conjuntas
        self.joint_tuples = list(product(*(self.A[j] for j in self.players)))
        # Agrupar tuplas por la acción propia para acelerar AV_i
        self.joint_by_action = {
            a_i: [joint for joint in self.joint_tuples if joint[self.idx] == a_i]
            for a_i in self.A[self.agent]
        }

        # Q[state][joint_tuple]
        self.Q = defaultdict(lambda: defaultdict(float))
        # Contadores de frecuencia de los demás agentes
        self.counts = {
            j: defaultdict(lambda: defaultdict(int))
            for j in self.players if j != self.agent
        }

    def procesar_observacion(self, obs):
        """
        Convierte una observación cruda en un estado hashable.
        - Numpy arrays se convierten a tuples.
        - Listas se convierten a tuples.
        - Otros tipos se devuelven tal cual (siempre que sean hashables).
        """
        if isinstance(obs, np.ndarray):
            return tuple(obs.tolist())
        if isinstance(obs, list):
            return tuple(obs)
        return obs

    def reset(self, epsilon: float = 1.0) -> None:
        """Reinicia para un nuevo episodio: estado inicial y ε."""
        self.epsilon     = epsilon
        self.last_state  = self.procesar_observacion(self.game.observe(self.agent))
        self.last_action = None

    def _pi_j(self, j, a_j, s):
        """Estimación de π̂_j(a_j|s) por frecuencia directa. Si no hay datos, uniforme."""
        cnts  = self.counts[j][s]
        total = sum(cnts.values())
        if total > 0:
            return cnts[a_j] / total
        return 1.0 / len(self.A[j])

    def AV_i(self, s, a_i):
        """Esperanza de Q bajo mi acción a_i, sumando solo tuplas relevantes."""
        v = 0.0
        for joint in self.joint_by_action[a_i]:
            p = 1.0
            for idx, j in enumerate(self.players):
                if idx == self.idx:
                    continue
                p *= self._pi_j(j, joint[idx], s)
            v += self.Q[s][joint] * p
        return v

    def action(self) -> int:
        """Epsilon-greedy sobre AV_i, actualiza exploring y last_action."""
        state = self.last_state
        if self.exploring and random.random() < self.epsilon:
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
        """Actualiza counts y Q usando la transición implícita de last_state/action."""
        s = self.last_state
        # convertir joint_action a tupla
        if isinstance(joint_action, dict):
            joint = tuple(joint_action[j] for j in self.players)
        else:
            joint = joint_action
        # recompensa y siguiente estado
        r       = self.game.reward(self.agent)
        raw_obs = self.game.observe(self.agent)
        s_next  = self.procesar_observacion(raw_obs)

        # actualizar contadores de los demás agentes
        for idx, j in enumerate(self.players):
            if j == self.agent:
                continue
            self.counts[j][s][joint[idx]] += 1

        # actualización Q-learning
        current_q = self.Q[s][joint]
        best_next = max(self.AV_i(s_next, a_i) for a_i in self.A[self.agent])
        target    = r + self.gamma * best_next
        self.Q[s][joint] = current_q + self.alpha * (target - current_q)

        # decaimiento epsilon si exploración
        if self.exploring:
            self.epsilon *= self.epsilon_decay

        # actualizar estado
        self.last_state = s_next
