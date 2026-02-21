import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.box_push_env import BoxPushEnv

class TestSinglePush(unittest.TestCase):
    def test_small_box_push(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "W B W",
            "W   W",
            "WWWWW"
        ]
        
        env = BoxPushEnv(ascii_map=ascii_map, render_mode=None)
        env.reset()
        
        # Agent should be at (2, 1) facing down (1). SmallBox at (2, 2).
        self.assertEqual(env.agent_pos, (2, 1))
        self.assertEqual(env.grid.get(2, 2).type, "box")
        
        # Action 2 is forward
        obs, reward, terminated, truncated, info = env.step(2)
        
        # Box should now be at (2, 3), agent at (2, 2)
        self.assertEqual(env.agent_pos, (2, 2))
        self.assertEqual(env.grid.get(2, 3).type, "box")
        self.assertIsNone(env.grid.get(2, 2)) # empty cell where box was
        
    def test_small_box_push_against_wall(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "W B W",
            "WWWWW"
        ]
        
        env = BoxPushEnv(ascii_map=ascii_map, render_mode=None)
        env.reset()
        
        self.assertEqual(env.agent_pos, (2, 1))
        
        # Agent pushes Box, but Box is against the Wall
        env.step(2)
        
        # Neither should move
        self.assertEqual(env.agent_pos, (2, 1))
        self.assertEqual(env.grid.get(2, 2).type, "box")

if __name__ == '__main__':
    unittest.main()
