import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from base.agent import Agent
from base.game import SimultaneousGame

def plot_policies(game: SimultaneousGame,
                  policies: ndarray,
                  action_labels: Optional[List[str]]=None,
                  it: Optional[int]=0
                  ) -> None:

    for agent in game.agents:
        plt.plot(policies[agent][:20], label = action_labels)
        plt.title(f'Iteración {it} - Evolución de la política de {agent} (ZOOM)')
        plt.legend()
        plt.show()

        plt.plot(policies[agent], label = action_labels)
        plt.title(f'Iteración {it} - Evolución de la política de {agent}')
        plt.legend()
        plt.show()

def plot_rewards(game: SimultaneousGame,
                 rewards: Dict[str, ndarray],
                 NITS: int=1,
                 agent_alias: Optional[Dict[str, str]]=None
                 ) -> None:
    _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for agent in game.agents:
        data = rewards[agent]
        mean = data.mean(axis=1)
        std = data.std(axis=1)

        data_cumsum = data.cumsum(axis=0)
        mean_cum = data_cumsum.mean(axis=1)
        std_cum = data_cumsum.std(axis=1)

        steps = np.arange(data.shape[0])
        label = agent_alias[agent] if agent_alias is not None else agent

        # Recompensas medias
        axes[0].plot(steps, mean, label=label)
        axes[0].fill_between(steps, mean - std, mean + std, alpha=0.3)

        # Recompensas acumuladas
        axes[1].plot(steps, mean_cum, label=label)
        axes[1].fill_between(steps, mean_cum - std_cum, mean_cum + std_cum, alpha=0.3)

    axes[0].set_title(f'Recompensas medias por paso - {NITS} iteraciones')
    axes[0].set_ylabel('Recompensa media')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title('Recompensas acumuladas')
    axes[1].set_xlabel('Paso')
    axes[1].set_ylabel('Recompensa acumulada media')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def iter_game(NITS: int,
              NSTEPS: int,
              game: SimultaneousGame,
              agent_classes: Dict[str, type[Agent]],
              action_labels: Optional[List[str]] = None,
              agent_alias: Optional[Dict[str, str]] = None,
              plot_simplex: bool = False
              ) -> None:

    rewards = {agent: np.zeros((NSTEPS, NITS)) for agent in game.agents}

    for it in range(NITS):
        pl = dict(map(lambda agent: (agent, agent_classes[agent](game=game, agent=agent)), game.agents))

        game.reset()
        print(f'Iteración {it+1}/{NITS}')

        policies = {agent: np.zeros((NSTEPS, game.action_spaces[agent].n)) for agent in game.agents}
        it_rewards = {agent: np.zeros(NSTEPS) for agent in game.agents}

        for i in range(NSTEPS):
            actions = dict(map(lambda agent: (agent, pl[agent].action()), game.agents))
            for agent in game.agents:
                policies[agent][i, :] = pl[agent].policy()
            _, step_rewards, _, _, _ = game.step(actions)
            for agent in game.agents:
                it_rewards[agent][i] = step_rewards[agent]

        for agent in game.agents:
            rewards[agent][:, it] = it_rewards[agent]

        plot_policies(game, policies, action_labels, it)

        print(f'Iteración {it+1} - {dict(map(lambda agent: (agent, pl[agent].policy()), game.agents))}')

    plot_rewards(game, rewards, NITS, agent_alias)

    return rewards, policies