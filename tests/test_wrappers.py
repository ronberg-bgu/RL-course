import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.multi_agent_env import MultiAgentBoxPushEnv
from environment.wrappers import StochasticActionWrapper, NoisyObservationWrapper

class TestWrappers(unittest.TestCase):
    def test_stochastic_wrapper(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "W   W",
            "W A W",
            "WWWWW"
        ]
        
        env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
        
        # 0% success wrappers - actions should never go through!
        wrapped_env = StochasticActionWrapper(env, p_success=0.0)
        wrapped_env.reset()
        
        # agent_0 is at (2,1)
        # agent_1 is at (2,3)
        self.assertEqual(wrapped_env.agent_positions["agent_0"], (2, 1))
        
        # Move forward
        actions = {"agent_0": 2, "agent_1": 2}
        wrapped_env.step(actions)
        
        # Since p=0, they should not have moved
        self.assertEqual(wrapped_env.agent_positions["agent_0"], (2, 1))
        
        # 100% success wrapper
        wrapped_env2 = StochasticActionWrapper(env, p_success=1.0)
        wrapped_env2.reset()
        wrapped_env2.step(actions)
        # They should have moved
        self.assertEqual(wrapped_env2.agent_positions["agent_0"], (2, 2))

if __name__ == '__main__':
    unittest.main()
