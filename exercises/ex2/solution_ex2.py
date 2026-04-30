"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop - DONE
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from itertools import product as iterproduct
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from llm_stochastic import run_online_replanning_episode, evaluate_online_replanning

# ---------------------------------------------------------------------------
# Map used in both parts (same as Assignment 1)
# ---------------------------------------------------------------------------
'''
ASCII_MAP = [
    "WWWWWW",
    "W A AW",
    "WBC BW",
    "WG GGW",
    "WWWWWW"
]
'''
ASCII_MAP = [
    "WWWWW",
    "WAA W",
    "WBBCW",
    "WGGGW",
    "WWWWW",
]


global transitions_model_singleton   # cache for build_transition_model, to avoid recomputing in MPI evaluation
transitions_model_singleton = None
# ===========================================================================
# Part 1 — Online Planning
# ===========================================================================

def run_online_planning(ascii_map: list, max_steps: int = 300, seed: int = None):
    """
    Thin wrapper around llm_stochastic.run_online_replanning_episode.

    Returns
    -------
    (success: bool, steps: int)
    """
    return run_online_replanning_episode(
        ascii_map=ascii_map,
        max_steps=max_steps,
        render=False,
        seed=seed,
    )


# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# Rotation is treated as free: agents are assumed to always face the right
# direction before moving, so agent direction is NOT part of the MDP state.
#
# A state is a tuple:
#   (agent0_pos, agent1_pos, box0_pos, box1_pos, heavy_pos)
#
# where positions are (col, row) tuples.
#
# Actions for the MDP are the four cardinal moves per agent:
#   0=right, 1=down, 2=left, 3=up
# Joint action = (act0, act1), one cardinal move per agent.
# "stay" is not modelled explicitly — a move into a wall simply fails.

def get_state(env) -> tuple:
    """
    Extract the current direction-free state tuple from a live environment.

    Uses env.box_positions (tracked by StochasticMultiAgentBoxPushEnv) so that
    agent sprites overwriting box cells in the grid cannot produce wrong results.
    """
    agents = env.possible_agents
    a0_pos = tuple(env.agent_positions[agents[0]])
    a1_pos = tuple(env.agent_positions[agents[1]])

    small_boxes = []
    heavy_boxes = []

    if hasattr(env, 'box_positions'):
        # Authoritative source: {id -> {"pos": (x,y), "size": "small"|"heavy"}}
        for info in env.box_positions.values():
            pos = tuple(info["pos"])
            if info["size"] == "heavy":
                heavy_boxes.append(pos)
            else:
                small_boxes.append(pos)
    else:
        for y in range(env.height):
            for x in range(env.width):
                cell = env.core_env.grid.get(x, y)
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        heavy_boxes.append((x, y))
                    else:
                        small_boxes.append((x, y))

    small_boxes.sort()
    heavy_boxes.sort()

    box0_pos  = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos  = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos = heavy_boxes[0] if heavy_boxes else None

    return (a0_pos, a1_pos, box0_pos, box1_pos, heavy_pos)


# ---------------------------------------------------------------------------
# Movement helpers
# ---------------------------------------------------------------------------

# Cardinal directions: index → (dx, dy)
CARDINAL = {
    0: ( 1,  0),   # right
    1: ( 0,  1),   # down
    2: (-1,  0),   # left
    3: ( 0, -1),   # up
}

ALL_CARDINAL = list(CARDINAL.keys())   # [0, 1, 2, 3]


def _is_wall(env, x: int, y: int) -> bool:
    """Return True if (x, y) is outside bounds or is a wall cell."""
    if x < 0 or y < 0 or x >= env.width or y >= env.height:
        return True
    cell = env.core_env.grid.get(x, y)
    return cell is not None and cell.type == "wall"


def build_transition_model(env: StochasticMultiAgentBoxPushEnv):
    """
    May god help us all. Build the transition model for the direction-free MDP."""
    global transitions_model_singleton
    if transitions_model_singleton is not None:
        return transitions_model_singleton
    from mpi import my_stoc_env
    my_env = my_stoc_env(env.ascii_map, move_success_prob=env.move_success_prob, push_success_prob=env.push_success_prob)
    #my_env.small_boxes_positions = [my_env.goal_positions[0], my_env.goal_positions[1]]
    #my_env.heavy_box_positions = [my_env.goal_positions[2]]
    states = [my_env.get_state()[0]]
    walls = my_env.get_state()[1]
    transitions = {}
    visited = set()
    while states:
        #print(f"State queue length : {len(states)} | Visited {len(visited)} states | Found {len(transitions)} transitions", end="\r")
        state = states.pop()
        visited.add(state)
        my_env.from_state(state, walls)
        for act0 in ALL_CARDINAL:
            for act1 in ALL_CARDINAL:
                joint_action = (act0, act1)
                transitions[(state, joint_action)] = my_env.get_transitions_full(act0, act1)
                for _, next_state, _ in transitions[(state, joint_action)]:
                    if next_state not in visited and next_state not in states:
                        states.append(next_state)
                        
    transitions_model_singleton = transitions, visited
    return transitions, visited
    
def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration (MPI) over the direction-free state space.

    Rotation is treated as free: the MDP state omits agent directions and
    actions are the four cardinal moves per agent. This reduces the state
    space by a factor of 16 (4 directions × 4 directions) compared to
    including orientation in the state.

    Algorithm per outer iteration
    -----------------------------
    1. Partial evaluation — k Bellman sweeps under the current policy:
           V(s) ← Σ_{s'} p(s'|s, π(s)) [r + γ V(s')]
    2. Policy improvement — greedy update:
           π(s) ← argmax_a Σ_{s'} p(s'|s, a) [r + γ V(s')]
    3. Convergence — stop when max_s |ΔV(s)| < theta.

    Returns
    -------
    policy : dict  state -> joint_action   (each action is (dir0, dir1))
    V      : dict  state -> float
    """
    transitions, states = build_transition_model(env)  # initialize V(s) = 0
    V = {s: 0.0 for s in states}
    policy = {s: (0, 0) for s in states}
    converged = False
    max_iters = 0
    while not converged and max_iters < max_outer_iters:
        # Partial evaluation
        for _ in range(k):
            V_new = {}
            for s in states:
                a0, a1 = policy[s]
                total = 0.0
                for p, s_next, r in transitions[(s, (a0, a1))]:
                    total += p * (r + gamma * V[s_next])
                V_new[s] = total
            V = V_new

        # Policy improvement
        policy_stable = True
        for s in states:
            old_action = policy[s]
            best_action = None
            best_value = float('-inf')
            for act0 in ALL_CARDINAL:
                for act1 in ALL_CARDINAL:
                    total = 0.0
                    for p, s_next, r in transitions[(s, (act0, act1))]:
                        total += p * (r + gamma * V[s_next])
                    if total > best_value:
                        best_value = total
                        best_action = (act0, act1)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        # Check convergence
        max_delta = max(abs(V[s] - sum(p * (r + gamma * V[s_next]) for p, s_next, r in transitions[(s, policy[s])]) ) for s in states)
        if max_delta < theta:
            converged = True

        max_iters += 1

    return policy, V
# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================

def evaluate_policy(policy_fn, env, n_runs: int = 100, max_steps: int = 500):
    """
    Run *policy_fn* for n_runs episodes and return (mean_steps, std_steps).

    Parameters
    ----------
    policy_fn : callable(env, obs) -> dict[agent -> action]
    env       : StochasticMultiAgentBoxPushEnv (reset inside each run)
    n_runs    : number of independent episodes
    max_steps : episode length cap (counts as a failure if hit)
    """
    steps_per_run = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        steps  = 0
        done   = False
        while not done and steps < max_steps:
            #print(env.ascii_map)
            actions = policy_fn(env, obs)
            obs, rewards, terms, truncs, _ = env.step(actions)
            steps += 1
            done = any(terms.values()) or any(truncs.values())

        steps_per_run.append(steps)

    return float(np.mean(steps_per_run)), float(np.std(steps_per_run))


# ===========================================================================
# Main — run both algorithms and print results
# ===========================================================================

if __name__ == "__main__":
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    # ── Part 1: Online Planning ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 - Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    stats = evaluate_online_replanning(ascii_map=ASCII_MAP, n_runs=100, max_steps=300)
    mean_ol = stats["mean_steps"]
    std_ol  = stats["std_steps"]
    print(f"  Success rate: {stats['successes']}/{stats['runs']} "
          f"({stats['success_rate']*100:.1f}%)")
    print(f"\nOnline Planning  ->  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 - Modified Policy Iteration")
    print("=" * 60)
    def flip_map(a_map:list[str])->list[str]:
        a_map[2] = a_map[2][::-1]
        print(a_map)
        return a_map

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=flip_map(ASCII_MAP), max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)
    #print(f"V = {V}\n")


    def evaluate_mpi_policy(policy, env, n_runs=100, max_steps=500):
        """
        Execute the MPI policy on the stochastic env, one cardinal step at a time.

        The MDP abstraction treats rotation as free, so each policy step is:
          1. Silently set both agents' directions to the policy's target cardinals
             (no env.step call, no step counted — rotation is free).
          2. Send ACTION_FORWARD for both agents (one env.step, one step counted).

        This is the correct way to evaluate a direction-free cardinal policy on
        the MiniGrid env. Using evaluate_policy directly would count rotation steps
        and do spurious policy lookups mid-rotation.
        """
        from mpi import my_stoc_env

        steps_list = []
        for run_idx in range(n_runs):
            m_env = my_stoc_env(env.ascii_map, move_success_prob=env.move_success_prob, push_success_prob=env.push_success_prob)
            steps = 0
            while steps < max_steps:
                state = m_env.get_state()[0]
                a0, a1 = policy[state]
                m_env.stochastic_step([a0, a1])  # forward step in the policy's cardinal directions
                steps += 1
                #print(steps)
                if m_env.is_goal_state()==1:
                    break
            steps_list.append(steps)
            #print(f"Run {run_idx+1}/{n_runs} completed: steps = {steps}, success = {m_env.is_goal_state()}")
        mean_steps = float(np.mean(steps_list))
        std_steps  = float(np.std(steps_list))
        return mean_steps, std_steps

    mean_mpi, std_mpi = evaluate_mpi_policy(policy, env_mpi, n_runs=100, max_steps= 20000)
    print(f"\nMPI              ->  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")
