import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from solution_ex2 import run_online_planning, ASCII_MAP

t0 = time.time()
env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=2000)
steps = run_online_planning(env, max_replans=500)
t1 = time.time()

print(f"Episode finished in {steps} env-steps, took {t1-t0:.1f} seconds")