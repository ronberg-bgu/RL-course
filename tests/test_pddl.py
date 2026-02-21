import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.multi_agent_env import MultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl

class TestPDDL(unittest.TestCase):
    def test_pddl_generation_and_solving(self):
        ascii_map = [
            "WWWWW",
            "W A W",
            "W   W",
            "W B W",
            "W   W",
            "W G W",
            "WWWWW"
        ]
        
        env = MultiAgentBoxPushEnv(ascii_map=ascii_map)
        env.reset()
        
        # Keep isolated to tmp directory to not spam the root
        pddl_folder = os.path.join(os.path.dirname(__file__), "tmp_pddl")
        domain_path, problem_path = generate_pddl_for_env(env, pddl_folder=pddl_folder)
        
        self.assertTrue(os.path.exists(domain_path))
        self.assertTrue(os.path.exists(problem_path))
        
        # Test unified planning solver
        plan = solve_pddl(domain_path, problem_path)
        
        self.assertIsNotNone(plan, "The PDDL solver could not find a plan for a basic map.")
        self.assertGreater(len(plan.actions), 0, "Plan should have at least 1 action.")

if __name__ == '__main__':
    unittest.main()
