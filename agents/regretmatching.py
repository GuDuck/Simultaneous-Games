import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
          self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()
        a = actions[self.agent]
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float)
        _, base_reward, _, _, _ = g.step(played_actions)
        base_reward = base_reward[self.agent].item()
        aux_played_actions = played_actions.copy()
        for action in range(self.game._num_actions):
            if action != a:
                aux_played_actions[self.agent] = action
                _, rewards, _, _, _ = g.step(aux_played_actions)
                u[action] = rewards[self.agent].item()
            else:
                u[action] = base_reward
        
        r = np.array([float(u[action]-base_reward) for action in range(self.game._num_actions)])
        return r

    
    def regret_matching(self):
        cum_regrets_sum = 0
        for action in range(self.game.num_actions(self.agent)):
            cum_regrets_sum += max(0, self.cum_regrets[action])

        for action in range(self.game.num_actions(self.agent)):
            if cum_regrets_sum <= 0:
                self.curr_policy[action] = 1/self.game.num_actions(self.agent)
            else:
                self.curr_policy[action] = max(0, self.cum_regrets[action]) / cum_regrets_sum
            self.sum_policy[action] += self.curr_policy[action]

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
           return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1)).item()
    
    def policy(self):
        return self.learned_policy
