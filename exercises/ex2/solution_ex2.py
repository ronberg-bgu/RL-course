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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
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
ASCII_SIMPLE = [
    "WWWWWW",
    "WA A W",
    "W BBCW",
    "W GGGW",
    "WWWWWW",
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
            # if(len(plan.actions)==1): #last action
            #     done = True
            #     break
            # if (env.goals_achieved ==3):
            #     done = True
            #     break
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

DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)} #East, South, West, North - clockwise

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
    raise NotImplementedError("TODO: implement build_transition_model")

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
import itertools
from collections import deque
import heapq
def heuristic(state, goals):
    # Box positions: index 4, 5 (small), 6 (heavy)
    boxes = [state[4], state[5], state[6]]
    boxes = [b for b in boxes if b is not None]
    
    unclaimed_goals = list(goals)
    total_h = 0
    #First match the heavy box
    heavy_box = state[6]
    if heavy_box and unclaimed_goals:
        dists = [abs(heavy_box[0]-g[0]) + abs(heavy_box[1]-g[1]) for g in unclaimed_goals]
        min_d = min(dists)
        total_h += min_d
        unclaimed_goals.pop(dists.index(min_d))
        
    # Match remaining small boxes to remaining goals
    for b in [state[4], state[5]]:
        if b and unclaimed_goals:
            dists = [abs(b[0]-g[0]) + abs(b[1]-g[1]) for g in unclaimed_goals]
            min_d = min(dists)
            total_h += min_d
            unclaimed_goals.pop(dists.index(min_d))
            
    return total_h


def build_transition_model(env, max_depth=250, max_states=1000000):
    env.reset()
    # Direction vectors for simulation
    VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    # Setup Map Information
    walls = set()
    goals = set()
    for y, row in enumerate(env.ascii_map):
        for x, char in enumerate(row):
            if char == 'W': walls.add((x, y))
            if char == 'G': goals.add((x, y))

    # Attempt to fix "np.int64" unreadable output
    def clean_state(s):
        def clean_item(item):
            if item is None: return None
            if isinstance(item, (list, tuple)): return tuple(int(x) if hasattr(x, 'item') else int(x) for x in item)
            return int(item) if hasattr(item, 'item') else int(item)
        return tuple(clean_item(x) for x in s)

    # Lightweight Deadlock check (defined outside loop for performance)
    def is_deadlock(box_pos, walls, goals):
        if box_pos in goals:
            return False 
        x, y = box_pos
        is_wall_N = (x, y - 1) in walls
        is_wall_S = (x, y + 1) in walls
        is_wall_W = (x - 1, y) in walls
        is_wall_E = (x + 1, y) in walls
        # Corner Deadlock only
        if (is_wall_N or is_wall_S) and (is_wall_W or is_wall_E):
            return True
        return False

    initial_state = clean_state(get_state(env))
    
    # Priority Queue stores: (f_score, depth, state)
    start_h = heuristic(initial_state, goals)
    queue = [(start_h, 0, initial_state)]
    
    transitions = {}
    visited = {initial_state: 0} 
    goals_reached = 0
    min_h = 100
    joint_actions = list(itertools.product([0, 1, 2], repeat=2))
    outcome_masks = [(True, True), (True, False), (False, True), (False, False)]

    while queue:
        if max_states and len(transitions) >= max_states:
            print(f"Reached max state cap: {max_states}. Stopping A* exploration.")
            break
            
        f, depth, curr_s = heapq.heappop(queue)
        # if depth >= max_depth:
        #     continue # Don't expand children if this path is already too long
        
        curr_s = clean_state(curr_s)
        
        # Skip if a shorter path already exists
        if curr_s in transitions:
            continue
            
        #Check if reached goal already
        active_boxes = {curr_s[4], curr_s[5], curr_s[6]} - {None}
        is_goal = len(goals) > 0 and goals.issubset(active_boxes)

        if is_goal:
            goals_reached += 1
            print(f"GOAL REACHED! Total goals mapped: {goals_reached}")
            #Reward is 0, cost is 0.
            transitions[curr_s] = {ja: [(1.0, curr_s, 0)] for ja in joint_actions}
            continue

        transitions[curr_s] = {}

        
        for ja in joint_actions:
            outcomes = []
            for mask in outcome_masks:
                
                a_pos = [list(curr_s[0]), list(curr_s[2])]
                a_dir = [curr_s[1], curr_s[3]]
                s_boxes = [list(curr_s[4]) if curr_s[4] else None, 
                           list(curr_s[5]) if curr_s[5] else None]
                h_box = list(curr_s[6]) if curr_s[6] else None
                was_push = [False, False]

                # Handle Rotations
                for i in range(2):
                    if mask[i]:
                        if ja[i] == 0: a_dir[i] = (a_dir[i] - 1) % 4 # left
                        elif ja[i] == 1: a_dir[i] = (a_dir[i] + 1) % 4 # right

                # Handle Movement and Cooperative Push
                heavy_pushed = False
                if all(mask) and ja[0] == 2 and ja[1] == 2:
                    d0, d1 = VECS[a_dir[0]], VECS[a_dir[1]]
                    t0 = [a_pos[0][0] + d0[0], a_pos[0][1] + d0[1]]
                    t1 = [a_pos[1][0] + d1[0], a_pos[1][1] + d1[1]]

                    # Joint push logic
                    if t0 == h_box and t1 == h_box and d0 == d1:
                        target_hb = [h_box[0] + d0[0], h_box[1] + d0[1]]
                        if tuple(target_hb) not in walls and target_hb not in s_boxes:
                            h_box = target_hb
                            a_pos[0], a_pos[1] = t0, t1
                            heavy_pushed = True
                            was_push = [True, True]

                # Handle Small Boxes and Normal Movement
                if not heavy_pushed:
                    for i in range(2):
                        if mask[i] and ja[i] == 2:
                            dx, dy = VECS[a_dir[i]]
                            target = [a_pos[i][0] + dx, a_pos[i][1] + dy]

                            # 1. Wall Check
                            if tuple(target) in walls:
                                continue

                            # 2. Heavy Box Check  - cannot push
                            if target == h_box:
                                continue

                            # 3. Small Box Check
                            if target in s_boxes:
                                b_idx = s_boxes.index(target)
                                b_next = [target[0] + dx, target[1] + dy]

                                # Path clear for small box?
                                if (tuple(b_next) not in walls and 
                                    b_next not in s_boxes and 
                                    b_next != h_box):
                                    s_boxes[b_idx] = b_next
                                    a_pos[i] = target
                                    was_push[i] = True
                                continue 

                            # 4. Final Position Update
                            a_pos[i] = target
                
                # Check for deadlocks
                dead = any(b and is_deadlock(tuple(b), walls, goals) for b in s_boxes)
                if h_box and is_deadlock(tuple(h_box), walls, goals): 
                    dead = True

                # Probability calculation
                p0 = env.push_success_prob if (ja[0] == 2 and was_push[0]) else env.move_success_prob
                p1 = env.push_success_prob if (ja[1] == 2 and was_push[1]) else env.move_success_prob
                prob = {(True,True): p0*p1, (True,False): p0*(1-p1), (False,True): (1-p0)*p1, (False,False): (1-p0)*(1-p1)}[mask]

                if prob > 0:
                    if dead:
                        outcomes.append((prob, "DEADLOCK", -100))
                        if "DEADLOCK" not in visited:
                            visited["DEADLOCK"] = 0
                            transitions["DEADLOCK"] = {a: [(1.0, "DEADLOCK", -100)] for a in joint_actions}
                    else:
                        sb_final = sorted([tuple(b) for b in s_boxes if b])
                        next_s = (tuple(a_pos[0]), a_dir[0], tuple(a_pos[1]), a_dir[1],
                                  sb_final[0] if len(sb_final) > 0 else None,
                                  sb_final[1] if len(sb_final) > 1 else None,
                                  tuple(h_box) if h_box else None)
                        
                        reward = -1
                        new_depth = depth + 1
                        
                        # Add to Priority Queue if new or shorter path found
                        if next_s not in visited or new_depth < visited[next_s]:
                            visited[next_s] = new_depth
                            h = heuristic(next_s, goals)
                            
                            if h < min_h:
                                min_h = h
                                print(f"New Closest State (h={h}): {next_s}") #debug print for larger map
                                
                            heapq.heappush(queue, (new_depth + h, new_depth, next_s))
                        
                        outcomes.append((prob, next_s, reward))
            
            transitions[curr_s][ja] = outcomes
            
        if len(transitions) % 10000 == 0:
            print(f"# of states: {len(transitions)}")

    return transitions

def modified_policy_iteration(env, gamma=0.99, k=10, theta=1e-3, max_outer_iters=20):
    # 1. Build the model
    transitions = build_transition_model(env)
    states = list(transitions.keys())
    # Pre-cache items to avoid dictionary overhead in the loops
    state_items = list(transitions.items()) 
    actions = list(itertools.product([0, 1, 2], repeat=2))
    start_state = get_state(env)
    # 2. Initialize V and Policy
    V = {s: -50.0 for s in states}
    policy = {s: actions[0] for s in states}
    for s, s_actions in state_items:
    # Check if ANY action in this state leads to itself with 0 reward (goal)
        for a in actions:
            for _, next_s, reward in s_actions[a]:
                if reward == 0 and next_s == s:
                    V[s] = 0.0
    print(f"Starting MPI with {len(states)} states...")

    for i in range(max_outer_iters):
        # Partial Policy Evaluation 
        for _ in range(k):
            max_diff = 0
            for s, s_actions in state_items: 
                old_v = V[s]
                a = policy[s]
                
                # Calculate V based on current policy
                new_v = sum(prob * (reward + gamma * V.get(next_s, -100.0)) 
                            for prob, next_s, reward in s_actions[a])
                
                V[s] = new_v
                diff = abs(old_v - new_v)
                if diff > max_diff: max_diff = diff
            
            if max_diff < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        for s, s_actions in state_items:
            old_action = policy[s]
            
            best_q = -float('inf')
            best_a = old_action
            
            for a in actions:
                current_q = sum(prob * (reward + gamma * V.get(next_s, -100.0)) 
                                for prob, next_s, reward in s_actions[a])
                
                # Permanent spin fix
                if current_q > best_q + 1e-7:
                    best_q = current_q
                    best_a = a
                elif a == (2, 2) and abs(current_q - best_q) < 1e-7:
                    best_a = a
            
            policy[s] = best_a
            if old_action != best_a:
                policy_stable = False
        
        # Monitor progress every 5 iterations
        if i % 5 == 0:
            print(f"Iteration {i}: V(start) = {V.get(start_state, -666):.2f}, Policy(start) = {policy.get(start_state)}")

        if policy_stable:
            print(f"Policy converged at iteration {i}")
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
            if steps <30:
                print(f"State #{steps}: {get_state(env)}")
                print(f"Policy suggests: {actions}")
                print()
                if get_state(env) not in policy:
                    print("WARNING: Start state not found in Policy Dictionary!")
            obs, rewards, terms, truncs, _ = env.step(actions)
            steps += 1
            done = any(terms.values()) or any(truncs.values())

        steps_per_run.append(steps)

    return float(np.mean(steps_per_run)), float(np.std(steps_per_run))


# ===========================================================================
# Main — run both algorithms and print results
# ===========================================================================
import time
if __name__ == "__main__":
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    # ── Part 1: Online Planning ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 — Online Planning (classical planner on stochastic env)")
    print("=" * 60)
    time_started = time.time()
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
    for i in range(20):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_SIMPLE, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 10 == 0:
            print(f"  run #{i+1};  steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    
    print(f"\nOnline Planning: mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")
    print(f"\nonline planning done in {time.time()-time_started}")
    time_started = time.time()
    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 — Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_SIMPLE, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi, gamma=0.99)
    
    def mpi_policy_fn(env, obs):
        """Convert current env state to a joint action using the MPI policy."""
        state = get_state(env)
        joint_action = policy.get(state, (0,0))
        # joint_action is a tuple (action_agent0, action_agent1)
        agents = env.possible_agents
        return {agents[0]: joint_action[0], agents[1]: joint_action[1]}
    print(V[get_state(env_mpi)])
    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=20)
    print(f"\nMPI              : mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")
    print(f"\nMPI planning done in {time.time()-time_started}")
    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")
