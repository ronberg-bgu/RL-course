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
import contextlib
import io

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

import unified_planning as up

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
    # Silence the Fast Downward credits stream to clean up the console
    up.shortcuts.get_environment().credits_stream = None

    obs, _ = env.reset()
    total_env_steps = 0
    done = False
    
    # Tracking for deadlock detection
    last_obs_str = None
    consecutive_failures = 0

    for _ in range(max_replans):
        if done:
            break

        # ── Deadlock Detection ───────────────────────────────────────
        current_obs_str = str(obs)
        if current_obs_str == last_obs_str:
            consecutive_failures += 1
            # If the state hasn't changed in 5 replans, we are deadlocked.
            # (Allows for a few standard 0.2 stochastic failures before aborting)
            if consecutive_failures >= 5:
                break 
        else:
            consecutive_failures = 0
            last_obs_str = current_obs_str

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

    # If we aborted due to deadlock or ran out of replans, apply the penalty
    if not done:
        return max_replans * 3
        
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

def get_canonical_state(env) -> tuple:
    """Extract the current state tuple containing ONLY sorted coordinates."""
    agents = env.possible_agents
    
    # 1. Get raw positions (NO DIRECTION!) and safely cast to tuple
    agent_positions = [
        tuple(env.agent_positions[agents[0]]), 
        tuple(env.agent_positions[agents[1]])
    ]
    
    # Sort them to make the agents interchangeable
    agents_tuple = tuple(sorted(agent_positions))

    # 2. Collect box positions
    small_boxes = []
    heavy_pos = None
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy_pos = (x, y)
                else:
                    small_boxes.append((x, y))

    # 3. Sort small boxes
    small_boxes.sort()
    boxes_tuple = tuple(small_boxes)

    # State is now incredibly compact!
    return (agents_tuple, boxes_tuple, heavy_pos)

def get_single_agent_intents(state, agent_index, action, env):
    """
    action is now an intended cardinal direction: 
    0: Right, 1: Down, 2: Left, 3: Up
    """
    agents_tuple, boxes_tuple, heavy_pos = state
    agent_pos = agents_tuple[agent_index]
    
    # The action IS the direction!
    target_vec = DIR_TO_VEC[action]
    target_pos = (agent_pos[0] + target_vec[0], agent_pos[1] + target_vec[1])

    # Build collision array
    box_positions = [pos for pos in boxes_tuple if pos is not None]
    if heavy_pos is not None:
        box_positions.append(heavy_pos)

    target_cell = env.core_env.grid.get(target_pos[0], target_pos[1])
    target_is_wall = (target_cell is not None and target_cell.type == "wall")

    # Case A: Wall
    if target_is_wall:
        return [(1.0, "stay")]
    
    # Case B: Box Interaction
    elif target_pos in box_positions:
        push_target_pos = (target_pos[0] + target_vec[0], target_pos[1] + target_vec[1])
        push_cell = env.core_env.grid.get(push_target_pos[0], push_target_pos[1])
        push_is_wall = (push_cell is not None and push_cell.type == "wall")

        if not (push_is_wall or (push_target_pos in box_positions)):
            # We must track the specific push direction for joint heavy pushes later
            return [(0.8, f"push_{action}"), (0.2, "stay")]
        else:
            return [(1.0, "stay")]
        
    # Case C: Move to Empty Space (The 0.8 / 0.1 / 0.1 logic)
    else:
        slip_left = (action - 1) % 4
        slip_right = (action + 1) % 4
        return [
            (0.8, f"move_{action}"), 
            (0.1, f"move_{slip_left}"), 
            (0.1, f"move_{slip_right}")
        ]
    
from minigrid.core.constants import DIR_TO_VEC


def simulate_joint_intents(state, joint_intent, env):
    # 1. Unpack the canonical state
    agents_tuple, boxes_tuple, heavy_pos = state
    a0_pos = agents_tuple[0]
    a1_pos = agents_tuple[1]

    # Helper function to decode the new action strings AND prevent wall/box clipping
    def parse_intent(pos, intent_str):
        if intent_str == "stay":
            return pos, None
        
        parts = intent_str.split("_")
        action_type = parts[0]
        direction = int(parts[1])
        
        vec = DIR_TO_VEC[direction]
        new_pos = (pos[0] + vec[0], pos[1] + vec[1])
        
        # SAFETY CHECK 1: Out of bounds (just in case)
        if not (0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height):
            return pos, None
            
        # SAFETY CHECK 2: Prevent slipping/walking into a wall
        cell = env.core_env.grid.get(new_pos[0], new_pos[1])
        if cell is not None and cell.type == "wall":
            return pos, None
            
        # SAFETY CHECK 3: Prevent slipping into a box
        if action_type == "move" and (new_pos in boxes_tuple or new_pos == heavy_pos):
            return pos, None
            
        push_target = None
        if action_type == "push":
            push_target = (new_pos[0] + vec[0], new_pos[1] + vec[1])
            
        return new_pos, push_target

    # 2. Calculate tentative moves
    new_a0_pos, a0_push_target = parse_intent(a0_pos, joint_intent[0])
    new_a1_pos, a1_push_target = parse_intent(a1_pos, joint_intent[1])

    new_heavy_pos = heavy_pos
    new_boxes = list(boxes_tuple) 

    # 3. Resolve Heavy Box Pushes
    if new_a0_pos == heavy_pos and new_a1_pos == heavy_pos:
        if a0_push_target == a1_push_target and a0_push_target is not None:
            new_heavy_pos = a0_push_target
        else:
            new_a0_pos, new_a1_pos = a0_pos, a1_pos 
            
    elif new_a0_pos == heavy_pos:
        new_a0_pos = a0_pos 
    elif new_a1_pos == heavy_pos:
        new_a1_pos = a1_pos 

    # 4. Resolve Small Box Pushes
    for i, b_pos in enumerate(boxes_tuple):
        if new_a0_pos == b_pos:
            new_boxes[i] = a0_push_target
        if new_a1_pos == b_pos:
            new_boxes[i] = a1_push_target

    # 5. Agent Collision Checks
    swapped = (new_a0_pos == a1_pos and new_a1_pos == a0_pos)
    same_cell = (new_a0_pos == new_a1_pos)
    box_hit_agent = (a0_push_target == new_a1_pos) or (a1_push_target == new_a0_pos)
    
    if swapped or same_cell or box_hit_agent:
        new_a0_pos, new_a1_pos = a0_pos, a1_pos
        new_boxes = list(boxes_tuple) 
        new_heavy_pos = heavy_pos     

    # 6. Box Collision Check
    if (a0_push_target is not None and a1_push_target is not None) and (a0_push_target == a1_push_target) and (new_heavy_pos == heavy_pos):
        new_a0_pos, new_a1_pos = a0_pos, a1_pos
        new_boxes = list(boxes_tuple)

    # 7. Package back into Canonical State
    final_agents = tuple(sorted([new_a0_pos, new_a1_pos]))
    final_boxes = tuple(sorted([b for b in new_boxes if b is not None]))
    final_state = (final_agents, final_boxes, new_heavy_pos)
    
    # 8. GOAL CHECK & REWARD ASSIGNMENT
    all_simulated_box_positions = list(final_boxes)
    if new_heavy_pos is not None:
        all_simulated_box_positions.append(new_heavy_pos)
        
    if set(all_simulated_box_positions) == set(env.goal_positions):
        reward = 1.0   
    else:
        reward = -1.0  

    return final_state, reward


import itertools

def build_transition_model(env):
    env.reset()

    transitions = {}
    
    # 1. Use our new directionless, sorted state extractor
    start_state = get_canonical_state(env)

    queue = [start_state]
    visited = set([start_state])

    # 2. Update to the Cardinal Direction action space!
    # 0: Right, 1: Down, 2: Left, 3: Up
    single_agent_actions = [0, 1, 2, 3]  
    
    # This generates 16 possible joint actions: (0,0), (0,1) ... (3,3)
    joint_actions = list(itertools.product(single_agent_actions, repeat=2))  
    
    while queue:
        current_state = queue.pop(0)
        transitions[current_state] = {}

        for joint_action in joint_actions:
            # joint_action[0] is the direction Agent 0 wants to go
            # joint_action[1] is the direction Agent 1 wants to go
            agent0_intents = get_single_agent_intents(current_state, 0, joint_action[0], env)
            agent1_intents = get_single_agent_intents(current_state, 1, joint_action[1], env)

            # Combine intents to get joint outcomes
            joint_outcomes = []
            
            for prob0, intent0 in agent0_intents:
                for prob1, intent1 in agent1_intents:
                    joint_prob = prob0 * prob1
                    joint_intent = (intent0, intent1)
                    
                    # 3. Simulate the physics (which now handles the "move_1", "stay", "push_3" strings)
                    # and returns the next_state already neatly sorted and canonical!
                    next_state, reward = simulate_joint_intents(current_state, joint_intent, env)
                    
                    joint_outcomes.append((joint_prob, next_state, reward))

            # 4. Collapse outcomes
            # Because of the 0.1 slip probabilities, it's possible for two entirely different 
            # combinations of slips to end up resulting in the exact same physical grid state.
            # We must sum their probabilities together so we don't have duplicate edges in our graph.
            collapsed_outcomes = {}
            for prob, next_state, reward in joint_outcomes:
                if next_state not in collapsed_outcomes:
                    collapsed_outcomes[next_state] = [0.0, reward]
                collapsed_outcomes[next_state][0] += prob
            
            final_outcomes = []
            for next_state, (prob, reward) in collapsed_outcomes.items():
                # Safety check: Only register transitions that actually have a chance of happening
                if prob > 0:
                    final_outcomes.append((prob, next_state, reward))
                    
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
            
            transitions[current_state][joint_action] = final_outcomes

    return transitions


import time

def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration for the Stochastic Box Pushing Environment.
    """
    # 1. Build the transition graph
    print("Building transition model (this might take a few seconds)...")
    transitions = build_transition_model(env)
    states = list(transitions.keys())
    print(f"State space size: {len(states)} states.")

    # 2. Initialization
    # Initialize V(s) = 0 for all states
    V = {s: 0.0 for s in states}
    
    # Initialize an arbitrary policy π(s)
    policy = {}
    for s in states:
        available_actions = list(transitions[s].keys())
        if available_actions:
            policy[s] = available_actions[0] # Just pick the first action
        else:
            policy[s] = None # Edge case: terminal state with no actions

    # Helper function to compute the Bellman expectation
    def expected_value(state, action):
        val = 0.0
        for prob, next_state, reward in transitions[state][action]:
            val += prob * (reward + gamma * V[next_state])
        return val

    # 3. The MPI Loop
    for iteration in range(max_outer_iters):
        iter_start_time = time.time() # <-- START TIMER
        
        # --- PHASE A: Partial Policy Evaluation ---
        # Evaluate the CURRENT policy for 'k' sweeps
        for _ in range(k):
            delta = 0.0
            new_V = {}
            for s in states:
                a = policy[s]
                if a is None: # Terminal state
                    new_V[s] = 0.0
                    continue
                
                v_new = expected_value(s, a)
                delta = max(delta, abs(v_new - V[s]))
                new_V[s] = v_new
            
            # Synchronous update
            V = new_V 
            
            # If the values stop changing significantly, break early to save time
            if delta < theta:
                break

        # --- PHASE B: Policy Improvement ---
        policy_stable = True
        
        for s in states:
            old_action = policy[s]
            if old_action is None:
                continue
            
            best_action = None
            best_value = -float('inf')
            
            # Test every possible action from this state to see if one is better
            for a in transitions[s].keys():
                val = expected_value(s, a)
                if val > best_value:
                    best_value = val
                    best_action = a
            
            # Update the policy with the newfound best action
            policy[s] = best_action
            
            # If the best action is different from what we used to do, the policy isn't stable yet
            if old_action != best_action:
                policy_stable = False
        
        # --- PHASE C: Convergence Check & Heartbeat ---
        iter_duration = time.time() - iter_start_time
        print(f"Iteration {iteration + 1} finished in {iter_duration:.2f}s | Max Delta: {delta:.4f}")

        if policy_stable:
            print(f"MPI converged successfully after {iteration + 1} outer iterations!")
            break
            
    else:
        print("Warning: MPI reached max_outer_iters without fully converging.")

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
        """Convert the MPI policy's cardinal directions into environment button presses."""
        # 1. Get the canonical state to look up the policy
        state = get_canonical_state(env)
        
        # 2. Extract the desired cardinal directions (e.g., 0=Right, 1=Down, 2=Left, 3=Up)
        # Note: Because canonical state sorts agents, we must ensure we map the action 
        # to the correct agent based on their sorted coordinates.
        agents = env.possible_agents
        # Update this line inside mpi_policy_fn:
        raw_positions = [
            tuple(env.agent_positions[agents[0]]), 
            tuple(env.agent_positions[agents[1]])
        ]
        
        # If Agent 1's coordinate was sorted to the front, we need to flip the actions!
        is_swapped = tuple(raw_positions) != tuple(sorted(raw_positions))
        
        canonical_action = policy[state]
        if is_swapped:
            desired_dir_0 = canonical_action[1]
            desired_dir_1 = canonical_action[0]
        else:
            desired_dir_0 = canonical_action[0]
            desired_dir_1 = canonical_action[1]

        # 3. Helper to translate desired compass direction -> button press
        def get_button_press(current_dir, desired_dir):
            if current_dir == desired_dir:
                return 2  # Already facing the right way -> Press "Forward"
            elif (current_dir - 1) % 4 == desired_dir:
                return 0  # Desired direction is to the left -> Press "Rotate Left"
            else:
                return 1  # Otherwise -> Press "Rotate Right"

        # 4. Generate the actual button presses
        action_agent0 = get_button_press(env.agent_dirs[agents[0]], desired_dir_0)
        action_agent1 = get_button_press(env.agent_dirs[agents[1]], desired_dir_1)

        return {agents[0]: action_agent0, agents[1]: action_agent1}

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    print(f"\nMPI              →  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60) 
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    # print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")