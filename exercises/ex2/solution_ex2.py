"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

from gettext import translation
import itertools
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions
from minigrid.core.constants import DIR_TO_VEC


# ---------------------------------------------------------------------------
# Map used in both parts (same as Assignment 1)
# ---------------------------------------------------------------------------
ASCII_MAP = [
    "WWWWWWWW",
    "W  AA  W",
    "W B C  W",
    "W      W",
    "W   B  W",
    "W G G GW",
    "WWWWWWWW",
]


# ===========================================================================
# Part 1 — Online Planning
# ===========================================================================

def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Execute one episode using online planning:
      replan from the current state → execute only the first PDDL action → repeat.

    Returns
    -------
    int
        Number of *env* steps taken (counting each rotate/forward individually).
        Returns max_replans * <average_actions_per_plan_step> as a large sentinel
        if the goal was never reached within max_replans replanning calls.
    """
    obs, _ = env.reset()
    total_env_steps = 0
    done = False

    for _ in range(max_replans):
        if done:
            break

        # ── 1. Export current state ──────────────────────────────────
        domain_path, problem_path = generate_pddl_for_env(env)

        # ── 2. Plan ──────────────────────────────────────────────────
        plan = solve_pddl(domain_path, problem_path)
        if not plan or len(plan.actions) == 0:
            break  # goal already reached (planner returns empty plan)

        # ── 3. Execute the first PDDL action ─────────────────────────
        pddl_action   = plan.actions[0]
        agent_targets = extract_target_pos(pddl_action)

        if not agent_targets:
            break

        # Build per-agent action queues (rotations + forward)
        agents_in_action = list(agent_targets.keys())
        action_queues = {
            a: get_required_actions(env, a, agent_targets[a])
            for a in agents_in_action
        }

        # Pad shorter queues so all agents execute their final forward together
        max_len = max(len(q) for q in action_queues.values())
        for a in agents_in_action:
            action_queues[a] = (
                [None] * (max_len - len(action_queues[a])) + action_queues[a]
            )

        # Step through the queue
        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {}
            for a in agents_in_action:
                if action_queues[a]:
                    act = action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act

            obs, rewards, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1

            if any(terms.values()) or any(truncs.values()):
                done = True
                break

    return total_env_steps


# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# A state is a tuple:
#   (agent0_pos, agent0_dir, agent1_pos, agent1_dir,
#    box0_pos,   box1_pos,   heavy_pos)
#
# where positions are (col, row) tuples and directions are 0-3.
#
# Feel free to simplify (e.g. drop agent directions if you argue they are
# irrelevant) as long as you justify it in your live demo.

def get_state(env) -> tuple:
    """Extract the current state tuple from a live environment."""
    agents = env.possible_agents
    a0_pos = env.agent_positions[agents[0]]
    a0_dir = env.agent_dirs[agents[0]]
    a1_pos = env.agent_positions[agents[1]]
    a1_dir = env.agent_dirs[agents[1]]

    # Collect box positions by scanning the grid
    small_boxes = []
    heavy_boxes = []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy_boxes.append((x, y))
                else:
                    small_boxes.append((x, y))

    # Sort for a canonical order
    small_boxes.sort()
    heavy_boxes.sort()

    box0_pos   = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos   = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos  = heavy_boxes[0] if heavy_boxes else None

    return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)

def get_single_agent_intents(state, agent_index, action, env):
    
    if action == 0:
        return [(1.0, "rotate_left")]
    elif action == 1:
        return [(1.0, "rotate_right")]
    
    if action == 2:

        if agent_index == 0:
            agent_pos = state[0]
            agent_dir = state[1]
        else:
            agent_pos = state[2]
            agent_dir = state[3]
        
        target_vec = DIR_TO_VEC[agent_dir]
        target_pos = (agent_pos[0] + target_vec[0], agent_pos[1] + target_vec[1])

        box_positions = [pos for pos in state[4:7] if pos is not None]

        target_cell = env.core_env.grid.get(target_pos[0], target_pos[1])
        target_is_wall = (target_cell is not None and target_cell.type == "wall")

        if target_is_wall:
            return [(1.0, "stay")]
        
        elif target_pos in box_positions:
            # Check if the box can be pushed
            push_target_pos = (target_pos[0] + target_vec[0], target_pos[1] + target_vec[1])
            push_cell = env.core_env.grid.get(push_target_pos[0], push_target_pos[1])
            push_is_wall = (push_cell is not None and push_cell.type == "wall")

            if not (push_is_wall or (push_target_pos in box_positions)):
                return [(0.8, "push"), (0.2, "stay")]
            else:
                return [(1.0, "stay")]
            
        else:
            return [(0.8, "move_forward"), (0.1, "slip_left"), (0.1, "slip_right")]
        
def simulate_joint_intents(state, joint_intent, env):
    a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos = state

    def get_tentative_move(pos, dir, intent):
        new_pos, new_dir = pos, dir
        push_box_target = None

        if intent == "rotate_left":
            new_dir = (dir - 1) % 4
        elif intent == "rotate_right":
            new_dir = (dir + 1) % 4
        elif intent == "move_forward" or intent == "push":
            target_vec = DIR_TO_VEC[dir]
            new_pos = (pos[0] + target_vec[0], pos[1] + target_vec[1])
            if intent == "push":
                push_box_target = (new_pos[0] + target_vec[0], new_pos[1] + target_vec[1])
        elif intent == "slip_left":
            slip_dir = (dir - 1) % 4
            slip_vec = DIR_TO_VEC[slip_dir]
            new_pos = (pos[0] + slip_vec[0], pos[1] + slip_vec[1])
        elif intent == "slip_right":
            slip_dir = (dir + 1) % 4
            slip_vec = DIR_TO_VEC[slip_dir]
            new_pos = (pos[0] + slip_vec[0], pos[1] + slip_vec[1])            

        return new_pos, new_dir, push_box_target
    
    new_a0_pos, new_a0_dir, a0_push_target = get_tentative_move(a0_pos, a0_dir, joint_intent[0])
    new_a1_pos, new_a1_dir, a1_push_target = get_tentative_move(a1_pos, a1_dir, joint_intent[1])

    new_box0_pos, new_box1_pos, new_heavy_pos = box0_pos, box1_pos, heavy_pos

    if (new_a0_pos == heavy_pos and new_a1_pos != heavy_pos) or (new_a1_pos == heavy_pos and new_a0_pos != heavy_pos):
        if new_a0_pos == heavy_pos: new_a0_pos = a0_pos
        if new_a1_pos == heavy_pos: new_a1_pos = a1_pos

    elif new_a0_pos == heavy_pos and new_a1_pos == heavy_pos:
        if a0_push_target == a1_push_target and a0_push_target is not None:
            new_heavy_pos = a0_push_target
        else:
            new_a0_pos, new_a1_pos = a0_pos, a1_pos

    if new_a0_pos == box0_pos: new_box0_pos = a0_push_target
    if new_a0_pos == box1_pos: new_box1_pos = a0_push_target
    if new_a1_pos == box0_pos: new_box0_pos = a1_push_target
    if new_a1_pos == box1_pos: new_box1_pos = a1_push_target

    if a0_push_target == new_a1_pos or a1_push_target == new_a0_pos:
        new_a0_pos, new_a1_pos = a0_pos, a1_pos
        new_box0_pos, new_box1_pos, new_heavy_pos = box0_pos, box1_pos, heavy_pos
        
    if (a0_push_target is not None and a1_push_target is not None) and (a0_push_target == a1_push_target) and (new_a0_pos != heavy_pos):
        new_a0_pos, new_a1_pos = a0_pos, a1_pos
        new_box0_pos, new_box1_pos = box0_pos, box1_pos 

    # Re-sort using the updated variable names
    small_boxes = [b for b in [new_box0_pos, new_box1_pos] if b is not None]
    small_boxes.sort()
    final_box0 = small_boxes[0] if len(small_boxes) > 0 else None
    final_box1 = small_boxes[1] if len(small_boxes) > 1 else None

    # Construct the final resulting state
    final_state = (new_a0_pos, new_a0_dir, new_a1_pos, new_a1_dir, final_box0, final_box1, new_heavy_pos)
    
    # Step reward is typically -1
    reward = -1 
    
    return final_state, reward
    

def build_transition_model(env):

    transitions = {}
    start_state = get_state(env)

    queue = [start_state]
    visited = set([start_state])

    single_agent_actions = [0,1,2]  # forward, rotate left, rotate right
    joint_actions = list(itertools.product(single_agent_actions, repeat=2))  # all joint actions
    
    while queue:
        current_state = queue.pop(0)
        transitions[current_state] = {}

        for joint_action in joint_actions:
            agent0_intents = get_single_agent_intents(current_state, 0, joint_action[0], env)
            agent1_intents = get_single_agent_intents(current_state, 1, joint_action[1], env)

            # Combine intents to get joint outcomes
            joint_outcomes = []
            
            for prob0, intent0 in agent0_intents:
                for prob1, intent1 in agent1_intents:
                    joint_prob = prob0 * prob1
                    joint_intent = (intent0, intent1)
                    next_state, reward = simulate_joint_intents(current_state, joint_intent, env)
                    joint_outcomes.append((joint_prob, next_state, reward))

            collapsed_outcomes = {}
            for prob, next_state, reward in joint_outcomes:
                if next_state not in collapsed_outcomes:
                    collapsed_outcomes[next_state] = [0.0, reward]
                collapsed_outcomes[next_state][0] += prob
            
            final_outcomes = []
            for next_state, (prob, reward) in collapsed_outcomes.items():
                final_outcomes.append((prob, next_state, reward))
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)
            
            transitions[current_state][joint_action] = final_outcomes

    return transitions




def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    TODO — Modified Policy Iteration.

    Parameters
    ----------
    env   : StochasticMultiAgentBoxPushEnv (used only to build the model)
    gamma : discount factor
    k     : number of partial policy-evaluation sweeps per iteration
    theta : convergence threshold for value change
    max_outer_iters : safety cap on outer iterations

    Returns
    -------
    policy : dict  state -> joint_action
    V      : dict  state -> float
    """
    raise NotImplementedError("TODO: implement modified_policy_iteration")


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
    print("Part 1 — Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    # Wrap run_online_planning as a policy function for the evaluator
    def online_planning_policy(env, obs):
        """
        This wrapper runs one COMPLETE episode internally and is only a shim
        for the evaluator.  evaluate_policy will reset the env before each
        call, so we hand control back immediately with a do-nothing action
        after the first step — the real logic is inside run_online_planning.

        NOTE: because run_online_planning drives the env loop itself, you
        should call it directly (see the manual loop below) for the 100-run
        evaluation; or adapt the evaluate_policy call to suit your design.
        """
        raise NotImplementedError(
            "Adapt this shim or call run_online_planning directly in a loop."
        )

    # Direct evaluation loop for online planning
    online_steps = []
    for i in range(100):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 10 == 0:
            print(f"  run {i+1}/100 — steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  →  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 — Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)

    def mpi_policy_fn(env, obs):
        """Convert current env state to a joint action using the MPI policy."""
        state = get_state(env)
        joint_action = policy[state]
        # joint_action is a tuple (action_agent0, action_agent1)
        agents = env.possible_agents
        return {agents[0]: joint_action[0], agents[1]: joint_action[1]}

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    print(f"\nMPI              →  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")