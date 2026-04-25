"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

import sys
import os
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

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

def parse_pddl_to_map(domain_file, problem_file):
    with open(problem_file, 'r') as f:
        problem_text = f.read()

    locations = re.findall(r'loc_(\d+)_(\d+)', problem_text)
    if not locations:
        raise ValueError("No locations found in the problem file.")
        
    # Visualizer expects loc_X_Y where X=Col, Y=Row
    max_x = max([int(x) for x, y in locations])
    max_y = max([int(y) for x, y in locations])

    # Arrays are accessed as grid[row][column] -> grid[y][x]
    grid = [[' ' for _ in range(max_x)] for _ in range(max_y)]

    # Grab everything between :init and :goal
    init_match = re.search(r'\(:init(.*?)\(:goal', problem_text, re.DOTALL | re.IGNORECASE)
    
    # FIX: Grab EVERYTHING from :goal to the end of the file
    goal_match = re.search(r'\(:goal(.*)', problem_text, re.DOTALL | re.IGNORECASE)
    
    init_text = init_match.group(1) if init_match else problem_text
    goal_text = goal_match.group(1) if goal_match else ""

    # Parse X (col) and Y (row) and place them correctly in the matrix
    for x, y in re.findall(r'loc_(\d+)_(\d+)', goal_text):
        grid[int(y)-1][int(x)-1] = 'G'
        
    for x, y in re.findall(r'agent-at\s+\S+\s+loc_(\d+)_(\d+)', init_text):
        grid[int(y)-1][int(x)-1] = 'A'
    for x, y in re.findall(r'box-at\s+\S+\s+loc_(\d+)_(\d+)', init_text):
        grid[int(y)-1][int(x)-1] = 'B'
    for x, y in re.findall(r'heavybox-at\s+\S+\s+loc_(\d+)_(\d+)', init_text):
        grid[int(y)-1][int(x)-1] = 'C'

    ascii_map = []
    wall_row = "W" * (max_x + 2)
    ascii_map.append(wall_row)
    for row in grid:
        ascii_map.append("W" + "".join(row) + "W")
    ascii_map.append(wall_row)

    return ascii_map

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
        x = parse_pddl_to_map(domain_path, problem_path)

        print("\nReconstructed ASCII Map:")
        for row in x:
            print(row)
        print("\n")

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

    #return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)
    return (a0_pos, a0_dir, box0_pos)

_static_grid_cache = None

def _get_static_grid(env):
    """Scan env grid once and cache wall/goal positions (they never change)."""
    global _static_grid_cache
    if _static_grid_cache is not None:
        return _static_grid_cache
    walls = set()
    goals = set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None:
                if cell.type == "wall":
                    walls.add((x, y))
                elif cell.type == "goal":
                    goals.add((x, y))
    _static_grid_cache = (frozenset(walls), frozenset(goals))
    return _static_grid_cache


def get_stochastic_outcomes(env, s, act) -> list:
    """
    Analytical transition model — single agent, single small box.
    s   = (agent_pos, agent_dir, box_pos)
          box_pos=None means the box is already on a goal (terminal state).
    act = 0 rotate-left | 1 rotate-right | 2 forward
    Returns [(probability, next_state, reward), ...]
    """
    agent_pos, agent_dir, box_pos = s
    move_p = env.move_success_prob
    push_p = env.push_success_prob
    side_p = (1.0 - move_p) / 2.0
    walls, goals = _get_static_grid(env)

    # Terminal — self-loop, no reward
    if box_pos is None:
        return [(1.0, s, 0.0)]

    # Rotate left / right — deterministic
    if act == 0:
        return [(1.0, (agent_pos, (agent_dir - 1) % 4, box_pos), 0.0)]
    if act == 1:
        return [(1.0, (agent_pos, (agent_dir + 1) % 4, box_pos), 0.0)]

    # Forward (action 2)
    dv  = DIR_TO_VEC[agent_dir]
    fwd = (agent_pos[0] + int(dv[0]), agent_pos[1] + int(dv[1]))

    # Wall directly ahead — no-op
    if fwd in walls:
        return [(1.0, s, 0.0)]

    # Push attempt (agent faces the box)
    if fwd == box_pos:
        dest = (box_pos[0] + int(dv[0]), box_pos[1] + int(dv[1]))
        if dest in walls:                       # precondition not met — no-op
            return [(1.0, s, 0.0)]
        reward  = 1.0 if dest in goals else 0.0
        new_box = None if dest in goals else dest
        return [
            (push_p,       (fwd, agent_dir, new_box), reward),
            (1.0 - push_p, s,                         0.0),
        ]

    # Move into free cell — three possible actual directions
    prob_map = {}
    for actual_dir, prob in [
        (agent_dir,            move_p),
        ((agent_dir - 1) % 4, side_p),
        ((agent_dir + 1) % 4, side_p),
    ]:
        av = DIR_TO_VEC[actual_dir]
        af = (agent_pos[0] + int(av[0]), agent_pos[1] + int(av[1]))
        # Deviated into wall or box → agent stays
        next_s = (af, agent_dir, box_pos) if (af not in walls and af != box_pos) else s
        prob_map[next_s] = prob_map.get(next_s, 0.0) + prob

    return [(p, ns, 0.0) for ns, p in prob_map.items()]

def build_transition_model(env):
    """
    TODO — Build the full MDP transition model analytically.

    This function should enumerate every reachable state and, for every state
    and every joint action, return the list of (probability, next_state, reward)
    triples that follow from the stochastic transition rules.

    Suggested signature of the returned data structure:

        transitions[state][joint_action] = [(prob, next_state, reward), ...]

    where joint_action is a tuple of per-agent actions, e.g. (2, 2) means
    both agents move forward simultaneously.

    Tips
    ----
    * Start with a *single*-agent, *single*-box toy map to validate your model
      before scaling to the full assignment map.
    * Use env.move_success_prob and env.push_success_prob for the probabilities.
    * A state is terminal if all boxes are at their goal positions — you can
      detect this by checking against the goal locations in the PDDL problem.
    """
    env.reset()
    initial_state = get_state(env)
    all_states = set()
    queue = [initial_state]
    transitions = {}
    
    # Define your joint actions here or pass them in
    actions = list(range(env.action_space(env.possible_agents[0]).n))
    #joint_actions = list(itertools.product(actions, actions))

    while queue:
        s = queue.pop(0)
        if s in all_states:
            continue
        all_states.add(s)
        
        transitions[s] = {}
        for act in actions:
            outcomes = get_stochastic_outcomes(env, s, act) 
            transitions[s][(act, 0)] = outcomes
            
            for prob, next_s, reward in outcomes:
                if next_s not in all_states:
                    queue.append(next_s)
                    
    return transitions # Crucial return for MPI to work


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
    # Get the model 
    transitions = build_transition_model(env)
    all_states = list(transitions.keys())
    
    # Intialize according to Readme
    V = {s: 0.0 for s in all_states}
    policy = {}
    for s in all_states:
        # First aritrary action
        first_action = next(iter(transitions[s].keys()))
        policy[s] = first_action

    # Outer Loop
    for outer_iter in range(max_outer_iters):
        
        # Partial policy evaluation
        for _ in range(k):
            delta = 0
            new_V = V.copy() 
            
            for s in all_states:
                action = policy[s]
                expected_value = 0.0
                
                # Bellman equation 
                for prob, next_s, reward in transitions[s][action]:
                    expected_value += prob * (reward + gamma * V[next_s])
                
                new_V[s] = expected_value
                delta = max(delta, abs(expected_value - V[s]))
            
            V = new_V
            
            if delta < theta:
                break

        # Policy improvement
        policy_stable = True
        
        for s in all_states:
            old_action = policy[s]
            best_action = None
            max_val = -float('inf')
            
            # Find the action that maximizes expected return: argmax_a Σ P * (R + γ * V(s'))
            for a, outcomes in transitions[s].items():
                val_a = 0.0
                for prob, next_s, reward in outcomes:
                    val_a += prob * (reward + gamma * V[next_s])
                
                if val_a > max_val:
                    max_val = val_a
                    best_action = a
            
            policy[s] = best_action
            
            # If the best action changed, our policy is not yet stable
            if best_action != old_action:
                policy_stable = False
                
        # Convergence check
        # If no actions changed for any state, we have found the optimal policy
        if policy_stable:
            print(f"MPI converged after {outer_iter + 1} iterations.")
            break

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
    for i in range(1):
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
