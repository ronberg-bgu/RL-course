import numpy as np
import random
from pettingzoo.utils.wrappers import BaseParallelWrapper

class StochasticActionWrapper(BaseParallelWrapper):
    """
    With probability p, an agent's action succeeds.
    With probability 1-p, it is ignored (agent slips and does nothing).
    """
    def __init__(self, env, p_success=0.9):
        super().__init__(env)
        self.p_success = p_success

    def step(self, actions):
        successful_actions = {}
        for agent, action in actions.items():
            if random.random() < self.p_success:
                successful_actions[agent] = action
        return super().step(successful_actions)

class NoisyObservationWrapper(BaseParallelWrapper):
    """
    Adds noise to the visual observation grid.
    With probability noise_level, an element in the (7x7x3) observation 'image' is replaced with a random integer.
    """
    def __init__(self, env, noise_level=0.05):
        super().__init__(env)
        self.noise_level = noise_level

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._add_noise(obs), info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)
        return self._add_noise(obs), rewards, terminations, truncations, infos
        
    def _add_noise(self, obs_dict):
        noisy_obs = {}
        for agent, agent_obs in obs_dict.items():
            # Copy observation space structure
            agent_obs_copy = dict(agent_obs)
            image = agent_obs_copy['image'].copy()
            
            # Create a mask of where to apply noise
            mask = np.random.random(image.shape) < self.noise_level
            
            # Minigrid image arrays are usually [0..10] max for standard types/colors/states
            random_noise = np.random.randint(0, 10, size=image.shape)
            
            # Apply
            image[mask] = random_noise[mask]
            agent_obs_copy['image'] = image
            noisy_obs[agent] = agent_obs_copy
            
        return noisy_obs
