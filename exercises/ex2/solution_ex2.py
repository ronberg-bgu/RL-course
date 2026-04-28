"""
Assignment 2 — Probabilistic Box Pushing
=========================================

This file implements and evaluates two distinct approaches to solving a stochastic
Multi-Agent Box Pushing environment.

Approach 1: Online Planning (Determinized Replanning)
-----------------------------------------------------
Model Used: Classical PDDL Planner (Fast Downward via Unified-Planning).
Code Flow:
1. The environment is paused at the current state.
2. `generate_pddl_for_env` dynamically generates a determinized `.pddl` representation
   of the grid, assuming actions (like pushing and moving) have a 100% success rate.
3. The classical planner finds an optimal path to the goal.
4. Because the environment is stochastic (e.g. 20% chance a push fails), we only execute 
   the very first step of the plan.
5. We observe the true outcome. If a slip or failure occurred, the next loop iteration 
   re-extracts the new state and replans. This creates a robust feedback loop.

Approach 2: Offline Solving via Modified Policy Iteration (MPI)
---------------------------------------------------------------
Model Used: Markov Decision Process (MDP) and Dynamic Programming.
Code Flow:
1. State Abstraction: To make the MDP tractable, we abstract the physical environment 
   into a simplified tuple: `(agent0_pos, agent1_pos, box0_pos, box1_pos, heavy_pos)`.
   Noticeably, Agent Direction is excluded. This reduces the state space by 16x.
2. Transition Model Generation (`build_transition_model`): We use Breadth-First Search (BFS) 
   to map out every reachable state and calculate exact transition probabilities 
   (0.8 for success, 0.1/0.1 for side deviations, 0.2 for push failures).
3. MPI Algorithm: We compute the optimal value function by alternating between `k=15` 
   sweeps of Partial Policy Evaluation (updating state values using the current policy) 
   and one sweep of Policy Improvement (greedily picking better actions based on the new values).
4. Policy Execution (`mpi_policy_fn`): During evaluation, since our abstract state doesn't
   track agent rotation, the policy function dynamically calculates and emits rotation 
   actions on-the-fly to ensure the agents face the right direction before moving.
"""

import sys
import os
import numpy as np
import pygame
import time

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

# Disable unified_planning credits for cleaner output
try:
    import unified_planning as up
    up.shortcuts.get_environment().credits_stream = None
except ImportError:
    pass

ASCII_MAP = [
    "WWWWWWWW",
    "W  AA  W",
    "W B C  W",
    "W      W",
    "W   B  W",
    "W G G GW",
    "WWWWWWWW",
]

def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Execute one episode using online planning.
    
    The Online Planner works by:
    1. Pausing the simulation at the current state.
    2. Extracting the current state into a deterministic PDDL domain and problem file.
    3. Running a classical planner (like Fast Downward) on the PDDL to find a plan to the goal.
       This assumes the "most likely" deterministic outcome will happen.
    4. Taking ONLY the first action from the generated plan.
    5. Executing that action in the stochastic environment.
    6. Repeating the process from step 1 with the newly observed state.
    
    Because the environment is stochastic, the actual outcome of the action might differ from 
    what the deterministic planner expected. By replanning at every step, the agent can recover 
    from unexpected outcomes or failures.
    """
    obs, _ = env.reset()
    total_env_steps = 0
    done = False

    for _ in range(max_replans):
        if done:
            break

        # ── 1. Export current state to PDDL ──────────────────────────
        # We use the pddl_extractor to scan the grid and create a text description
        # of the current objects, locations, and goal states.
        domain_path, problem_path = generate_pddl_for_env(env)

        # ── 2. Plan ──────────────────────────────────────────────────
        # solve_pddl calls the classical planning backend.
        plan = solve_pddl(domain_path, problem_path)
        
        # If the plan is empty, we have likely reached the goal already
        if not plan or len(plan.actions) == 0:
            break

        # ── 3. Execute the first PDDL action ─────────────────────────
        # In a stochastic environment, we only execute the very first step of the plan.
        pddl_action = plan.actions[0]
        agent_targets = extract_target_pos(pddl_action)

        if not agent_targets:
            break

        agents_in_action = list(agent_targets.keys())
        try:
            # We convert the target positions into actual low-level primitive actions
            # required by the environment (e.g., [Turn Right, Turn Right, Forward]).
            action_queues = {
                a: get_required_actions(env, a, agent_targets[a])
                for a in agents_in_action
            }
        except ValueError:
            # Fallback if the planner returns something unexpected
            obs, rewards, terms, truncs, _ = env.step({})
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            continue

        # We pad the action queues so that agents execute their final 'Forward' movement
        # simultaneously. This is critical for actions like pushing the heavy box together.
        max_len = max(len(q) for q in action_queues.values())
        for a in agents_in_action:
            action_queues[a] = (
                [None] * (max_len - len(action_queues[a])) + action_queues[a]
            )

        # Step through the queue until all actions for this PDDL step are executed
        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {}
            for a in agents_in_action:
                if action_queues[a]:
                    act = action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act

            obs, rewards, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1

            if env.render_mode == "human":
                env.core_env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                time.sleep(0.1)

            if any(terms.values()) or any(truncs.values()):
                done = True
                break

    return total_env_steps

# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

def get_state(env) -> tuple:
    """
    Extract the current abstract state from the environment.
    
    Tractability Choice: We drop agent directions from the state representation.
    Why? The agent can always rotate in place. While rotation takes a step, the core 
    complexity and bottlenecks are the positional changes and box pushing.
    By dropping direction, we reduce the state space by a factor of 16 (4 directions 
    per agent), making the MDP drastically smaller and faster to solve.
    Our execution policy will simply insert rotation actions dynamically when the 
    agent needs to move in a direction it's not facing.
    
    State tuple: (agent0_pos, agent1_pos, box0_pos, box1_pos, heavy_pos)
    """
    agents = env.possible_agents
    a0_pos = env.agent_positions[agents[0]]
    a1_pos = env.agent_positions[agents[1]]

    # Sort agents to halve the state space
    a_list = [a0_pos, a1_pos]
    a_list.sort()

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

    small_boxes.sort()
    heavy_boxes.sort()

    box0_pos   = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos   = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos  = heavy_boxes[0] if heavy_boxes else None

    return (a_list[0], a_list[1], box0_pos, box1_pos, heavy_pos)

def build_transition_model(env):
    """
    Builds the MDP transition model dynamically using BFS from the initial state.
    
    Returns a nested dictionary representing the MDP Physics:
    state -> { joint_action -> [(probability, next_state, reward), ...] }
    
    Physics Rules modeled here:
    - Moving to an empty cell: 80% chance of intended direction, 10% chance of slipping left, 
      10% chance of slipping right. If a slip targets a wall or a box, the agent stays put.
    - Pushing a box: 80% chance of success (box moves, agent follows), 20% chance of failure (both stay put).
    - Heavy boxes: Require both agents to execute the exact same push action simultaneously.
    
    Joint action is a tuple (act0_dir, act1_dir) where each is 0=right, 1=down, 2=left, 3=up.
    """
    move_succ = getattr(env, "move_success_prob", 0.8)
    push_succ = getattr(env, "push_success_prob", 0.8)
    side_prob = (1.0 - move_succ) / 2.0
    
    walkable = set()
    goals = set()
    env.reset()
    for y in range(env.height):
        for x in range(env.width):
            c = env.core_env.grid.get(x, y)
            if c is None or c.type != "wall":
                walkable.add((x, y))
            if c is not None and c.type == "goal":
                goals.add((x, y))
                
    import collections
    init_state = get_state(env)
    queue = collections.deque([init_state])
    visited = set([init_state])
    transitions = {}
    
    # 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
    single_actions = [0, 1, 2, 3] 
    joint_actions = [(a0, a1) for a0 in single_actions for a1 in single_actions]
    macro_vecs = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    
    def is_trans_terminal(state):
        if state == "FAIL": return False
        # We consider a state terminal if all boxes that exist are on goal cells
        cur_boxes = [b for b in state[2:] if b is not None]
        return len(cur_boxes) > 0 and all(b in goals for b in cur_boxes)
        
    while queue:
        state = queue.popleft()
        if state in transitions:
            continue
        transitions[state] = {}
        
        if len(transitions) % 1000 == 0:
            print(f"Processed {len(transitions)} states, queue size: {len(queue)}")
            
        if state == "FAIL":
            for ja in joint_actions:
                transitions[state][ja] = [(1.0, "FAIL", 0.0)]
            continue
        
        if is_trans_terminal(state):
            # Terminal states loop back to themselves with 0 reward
            for ja in joint_actions:
                transitions[state][ja] = [(1.0, state, 0.0)]
            continue
            
        a0_p, a1_p, b0_p, b1_p, h_p = state
        boxes = {}
        if b0_p: boxes[b0_p] = "small"
        if b1_p: boxes[b1_p] = "small"
        if h_p: boxes[h_p] = "heavy"
        
        for ja in joint_actions:
            act0, act1 = ja
            intents = {0: None, 1: None}
            
            for idx, (p, act_dir) in enumerate([(a0_p, act0), (a1_p, act1)]):
                vec = macro_vecs[act_dir]
                fwd = (p[0]+vec[0], p[1]+vec[1])
                intents[idx] = {"tgt": fwd, "dir": act_dir, "vec": vec}
            
            def get_branches(intent, p):
                tgt = intent["tgt"]
                box_type = boxes.get(tgt)
                
                # Check bounds and walls. If target is out of walkable bounds, it's a fail (stay in place)
                if tgt not in walkable:
                    return [(1.0, "fail")]
                    
                if box_type in ["small", "heavy"]:
                    # Attempting to push. Success -> push, Fail -> stay in place.
                    # We flag it as a 'push' intent so the resolution logic knows to push.
                    intent["is_push"] = True
                    return [(push_succ, intent), (1.0 - push_succ, "fail")]
                
                # Moving into empty space
                intent["is_push"] = False
                d = intent["dir"]
                dl = (d - 1) % 4; v_l = macro_vecs[dl]; tl = (p[0]+v_l[0], p[1]+v_l[1])
                dr = (d + 1) % 4; v_r = macro_vecs[dr]; tr = (p[0]+v_r[0], p[1]+v_r[1])
                
                branches = []
                branches.append((move_succ, {"tgt": tgt, "vec": intent["vec"], "is_push": False} if tgt in walkable else "fail"))
                branches.append((side_prob, {"tgt": tl, "vec": v_l, "is_push": False} if tl in walkable else "fail"))
                branches.append((side_prob, {"tgt": tr, "vec": v_r, "is_push": False} if tr in walkable else "fail"))
                return branches
            
            branches0 = get_branches(intents[0], a0_p)
            branches1 = get_branches(intents[1], a1_p)
            
            is_joint_heavy = False
            # Check if both agents are jointly pushing the heavy box
            if (intents[0]["tgt"] == intents[1]["tgt"] and 
                boxes.get(intents[0]["tgt"]) == "heavy" and 
                a0_p == a1_p and 
                intents[0]["dir"] == intents[1]["dir"]):
                is_joint_heavy = True
            
            outcomes = {}
            if is_joint_heavy:
                # Heavy push resolves as a single joint event
                joint_branches = [(push_succ, (intents[0], intents[1])), (1.0 - push_succ, ("fail", "fail"))]
            else:
                # Independent movements
                joint_branches = [(p0 * p1, (b0, b1)) for p0, b0 in branches0 for p1, b1 in branches1]
            
            for prob, (b0_i, b1_i) in joint_branches:
                if prob == 0: continue
                new_b0 = b0_p; new_b1 = b1_p; new_h = h_p
                na0_p = a0_p; na1_p = a1_p
                nbs = dict(boxes)
                heavy_consumed = {0: False, 1: False}
                
                if is_joint_heavy and b0_i != "fail":
                    hv_tgt = intents[0]["tgt"]
                    nhv = (hv_tgt[0]+intents[0]["vec"][0], hv_tgt[1]+intents[0]["vec"][1])
                    if nhv in walkable and nbs.get(nhv) is None:
                        del nbs[hv_tgt]; nbs[nhv] = "heavy"
                        new_h = nhv; na0_p = hv_tgt; na1_p = hv_tgt
                    heavy_consumed[0] = True; heavy_consumed[1] = True
                        
                for idx, (b_intent, a_p, consumed) in enumerate([(b0_i, a0_p, heavy_consumed[0]), (b1_i, a1_p, heavy_consumed[1])]):
                    if consumed or b_intent == "fail": continue
                    
                    tgt = b_intent["tgt"]
                    vec = b_intent["vec"]
                    
                    if nbs.get(tgt) == "small":
                        # If we deviated laterally into a box, we don't push it; we stay put.
                        if not b_intent.get("is_push", False):
                            pass 
                        else:
                            # Pushing small box
                            ntgt = (tgt[0]+vec[0], tgt[1]+vec[1])
                            if ntgt in walkable and nbs.get(ntgt) is None:
                                if b0_p == tgt: new_b0 = ntgt
                                elif b1_p == tgt: new_b1 = ntgt
                                del nbs[tgt]; nbs[ntgt] = "small"
                                if idx == 0: na0_p = tgt
                                else: na1_p = tgt
                    elif nbs.get(tgt) == "heavy":
                        # Failed to push heavy because not joint, or deviated into it -> acts as wall, stay in place
                        pass 
                    elif nbs.get(tgt) is None:
                        # Moving into empty space
                        if idx == 0: na0_p = tgt
                        else: na1_p = tgt
                        
                ns = (na0_p, na1_p, new_b0, new_b1, new_h)
                
                # IMPORTANT: Normalize agents and small boxes so state representation is unique
                a_list = [na0_p, na1_p]
                a_list.sort()
                
                small_list = []
                if ns[2]: small_list.append(ns[2])
                if ns[3]: small_list.append(ns[3])
                small_list.sort()
                nb0 = small_list[0] if len(small_list) > 0 else None
                nb1 = small_list[1] if len(small_list) > 1 else None
                
                # Prune Dead Ends: If any box is against the top wall (y=1) or left wall (x=1),
                # it can NEVER reach a goal because an agent cannot get behind it to push it.
                is_dead = False
                for bx in [nb0, nb1, ns[4]]:
                    if bx is not None and (bx[0] == 1 or bx[1] == 1):
                        is_dead = True
                        break
                
                if is_dead:
                    ns = "FAIL"
                else:
                    ns = (a_list[0], a_list[1], nb0, nb1, ns[4])
                
                outcomes[ns] = outcomes.get(ns, 0.0) + prob
                
            res_list = []
            for ns, prob in outcomes.items():
                rew = 1.0 if is_trans_terminal(ns) else 0.0
                res_list.append((prob, ns, rew))
                if ns not in visited: 
                    visited.add(ns)
                    queue.append(ns)
                    
            transitions[state][ja] = res_list
            
    return transitions


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 15,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration (MPI) Algorithm.
    
    MPI is a variant of Policy Iteration. Instead of evaluating a policy to full convergence 
    at every step (which is computationally expensive), it performs exactly 'k' sweeps of 
    value updates (Partial Policy Evaluation) before doing a Policy Improvement step.
    
    This strikes a balance between Value Iteration (k=1) and full Policy Iteration (k=infinity),
    often leading to faster convergence in practice.
    """
    print("Building transition model...")
    transitions = build_transition_model(env)
    print(f"Model built! {len(transitions)} reachable states.")
    
    states = list(transitions.keys())
    V = {s: 0.0 for s in states}
    policy = {s: (0, 0) for s in states}
    
    # Initialize policy arbitrarily (just pick the first available action)
    for s in states:
        policy[s] = next(iter(transitions[s].keys()))
        
    for i in range(max_outer_iters):
        # 1. Partial policy evaluation (k sweeps)
        # We update the value function V(s) based on the CURRENT policy.
        # We do this 'k' times, which gives a good enough estimate to improve the policy.
        for _ in range(k):
            V_new = {}
            for s in states:
                act = policy[s]
                v_s = 0.0
                for prob, next_s, reward in transitions[s][act]:
                    v_s += prob * (reward + gamma * V[next_s])
                V_new[s] = v_s
            V = V_new
                
        # 2. Policy improvement
        # We now look for a better action for each state, assuming our updated V(s) is true.
        policy_stable = True
        for s in states:
            old_act = policy[s]
            best_act = None
            best_value = -float('inf')
            
            for act in transitions[s].keys():
                val = 0.0
                for prob, next_s, reward in transitions[s][act]:
                    val += prob * (reward + gamma * V[next_s])
                if val > best_value:
                    best_value = val
                    best_act = act
                    
            policy[s] = best_act
            
            # If any state changes its preferred action, the policy has not converged.
            if old_act != best_act:
                policy_stable = False
                
        if policy_stable:
            print(f"MPI converged after {i+1} outer iterations.")
            break
            
    return policy, V

# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================

def evaluate_policy(policy_fn, env, n_runs: int = 100, max_steps: int = 500):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assignment 2 Solution")
    parser.add_argument("--visualize", action="store_true", help="Enable PyGame visualization")
    parser.add_argument("--collect-stats", type=int, default=1, metavar="RUNS_NUMBER", help="Run RUNS_NUMBER evaluations to collect stats (default 1)")
    args = parser.parse_args()

    render_mode = "human" if args.visualize else None
    
    n_runs_online = args.collect_stats
    n_runs_mpi = args.collect_stats

    # ── Part 1: Online Planning ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 — Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    online_steps = []
    print(f"Running {n_runs_online} evaluations for online planner (this will take time)...")
    for i in range(n_runs_online):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500, render_mode=render_mode)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 5 == 0 or args.visualize:
            print(f"  run {i+1}/{n_runs_online} — steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  →  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 — Modified Policy Iteration")
    print("=" * 60)

    # Transition model doesn't need to be rendered
    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)

    def mpi_policy_fn(env, obs):
        """
        Converts the current physical environment state into primitive joint actions 
        based on the offline-computed MPI policy.
        
        Because our MDP state abstraction ignored agent rotation (for tractability), 
        this function must act as a 'controller' to bridge the gap. It translates 
        the abstract macro-intention (e.g., "Move UP") into the actual sequence of 
        primitives required by MiniGrid (e.g., "Turn Left, Turn Left, Forward").
        """
        # 1. Map current physical state to abstract MDP state
        state = get_state(env)
        
        # 2. Query policy for the best abstract joint action
        if state not in policy:
            return {env.possible_agents[0]: 0, env.possible_agents[1]: 0}
            
        joint_action = policy[state] # (dir0, dir1) where 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        
        # 3. Translate abstract direction intention into physical actions
        # Because we abstracted agent orientation away and sorted their positions, 
        # we must map the joint action back to the physical agents by sorting them 
        # the exact same way they were sorted in `get_state`.
        sorted_agents = sorted(env.possible_agents, key=lambda a: env.agent_positions[a])
        out_acts = {}
        
        # MiniGrid Directions: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        for idx, agent in enumerate(sorted_agents):
            target_dir = joint_action[idx]
            cur_dir = env.agent_dirs[agent]
            
            if cur_dir == target_dir:
                # Facing the right way, execute forward step
                out_acts[agent] = 2 
            else:
                # Need to rotate. Calculate the optimal rotation direction.
                diff = (target_dir - cur_dir) % 4
                if diff == 1:
                    out_acts[agent] = 1 # turn right
                elif diff == 3:
                    out_acts[agent] = 0 # turn left
                else:
                    # Target is 180 degrees behind. Just turn right for now.
                    # It will take 2 steps to fully turn around.
                    out_acts[agent] = 1 
                
        return out_acts

    print(f"Running {n_runs_mpi} evaluations for MPI planner...")
    env_mpi_eval = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500, render_mode=render_mode)
    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi_eval, n_runs=n_runs_mpi)
    print(f"\nMPI              →  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = (
        "=" * 60 + "\n" +
        "SUMMARY\n" +
        "=" * 60 + "\n" +
        f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}\n" +
        "-" * 50 + "\n" +
        f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}\n" +
        f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}\n"
    )
    
    print(summary)
    
    if args.collect_stats:
        results_path = os.path.join(os.path.dirname(__file__), "results.txt")
        with open(results_path, "w") as f:
            f.write(summary)
        print(f"Results successfully written to {results_path}")
