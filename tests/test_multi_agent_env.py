import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.multi_agent_env import MultiAgentBoxPushEnv

class TestMultiAgentAPI(unittest.TestCase):
    def test_multi_agent_parallel_api(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "W B W",
            "W A W",
            "WWWWW"
        ]
        
        env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
        obs, info = env.reset()
        
        self.assertIn("agent_0", obs)
        self.assertIn("agent_1", obs)
        self.assertEqual(len(env.agents), 2)
        
        # agent_0 is at (2, 1) facing down, small box at (2, 2). Action 2 pushes it.
        # agent_1 is at (2, 3). Action 0 rotates left.
        actions = {
            "agent_0": 2,
            "agent_1": 0
        }
        
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        self.assertIn("agent_0", next_obs)
        self.assertIn("agent_1", next_obs)
        self.assertFalse(terminations["agent_0"])
        self.assertFalse(truncations["agent_0"])
        
        # Validate grid state
        # agent_1 was at (2,3) blocking the box, so agent_0's push should fail!
        # Thus, agent_0 should still be at (2, 1) and box at (2, 2)
        self.assertEqual(env.agent_positions["agent_0"], (2, 1))
        
    def test_multi_agent_successful_push(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "W B W",
            "W   W",
            "W A W",
            "WWWWW"
        ]
        
        env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
        env.reset()
        
        # agent_0 at (2, 1), box at (2, 2), empty at (2, 3)
        actions = {"agent_0": 2, "agent_1": 0}
        env.step(actions)
        
        # agent_0 should move to (2, 2), box to (2, 3)
        self.assertEqual(env.agent_positions["agent_0"], (2, 2))
        self.assertEqual(env.core_env.grid.get(2, 3).type, "box")

if __name__ == '__main__':
    unittest.main()
