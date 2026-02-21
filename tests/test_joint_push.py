import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.multi_agent_env import MultiAgentBoxPushEnv

class TestJointPush(unittest.TestCase):
    def test_single_agent_push_fails(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "WCC W", # BigBox spans 2 cells
            "W A W",
            "WWWWW"
        ]
        
        env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
        env.reset()
        
        # agent_0 is at (2, 1) facing down. Tries to push C at (2, 2)
        # agent_1 is at (2, 3) facing down. Rotates right.
        actions = {"agent_0": 2, "agent_1": 1}
        env.step(actions)
        
        # Should NOT move big box
        self.assertEqual(env.agent_positions["agent_0"], (2, 1))
        self.assertEqual(env.core_env.grid.get(2, 2).type, "box")
        self.assertEqual(getattr(env.core_env.grid.get(2, 2), "box_size", ""), "big")

    def test_joint_push_succeeds(self):
        ascii_map = [
            "WWWWWW",
            "W AA W", # agent_0 at (2,1), agent_1 at (3,1)
            "W CC W", # bigbox at (2,2) and (3,2)
            "W    W",
            "WWWWWW"
        ]
        
        env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
        env.reset()
        
        # Both push forward (down)
        # Verify agents
        self.assertEqual(env.agent_positions["agent_0"], (2, 1))
        self.assertEqual(env.agent_positions["agent_1"], (3, 1))
        
        actions = {"agent_0": 2, "agent_1": 2} # both forward
        env.step(actions)
        
        # Big Box should move down to y=3
        self.assertEqual(env.agent_positions["agent_0"], (2, 2))
        self.assertEqual(env.agent_positions["agent_1"], (3, 2))
        self.assertEqual(getattr(env.core_env.grid.get(2, 3), "box_size", ""), "big")
        self.assertEqual(getattr(env.core_env.grid.get(3, 3), "box_size", ""), "big")
        # Ensure previous space is now agents
        self.assertEqual(env.core_env.grid.get(2, 2).type, "agent")

if __name__ == '__main__':
    unittest.main()
