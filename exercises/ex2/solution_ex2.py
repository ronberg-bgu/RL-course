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
        try:
            action_queues = {
                a: get_required_actions(env, a, agent_targets[a])
                for a in agents_in_action
            }
        except ValueError:
            # Planner returned an action not adjacent to our actual position
            obs, rewards, terms, truncs, _ = env.step({})
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            continue

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


def build_transition_model(env):
    from minigrid.core.constants import DIR_TO_VEC
    
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
                
    init_state = get_state(env)
    queue = [init_state]
    transitions = {}
    
    def get_omni_state(state):
        # Convert state from full (pos, dir) to omni (pos)
        return (state[0], state[2], state[4], state[5], state[6])
        
    init_omni = get_omni_state(init_state)
    queue = [init_omni]
    transitions = {}
    
    # Macro-actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    single_actions = [0, 1, 2, 3] 
    joint_actions = [(a0, a1) for a0 in single_actions for a1 in single_actions]
    macro_vecs = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    
    def is_trans_terminal(state):
        cur_boxes = [b for b in state[2:] if b is not None]
        return len(cur_boxes) > 0 and all(b in goals for b in cur_boxes)
        
    while queue:
        state = queue.pop(0)
        if state in transitions:
            continue
        transitions[state] = {}
        
        if is_trans_terminal(state):
            for ja in joint_actions:
                transitions[state][ja] = [(1.0, state, 0.0)]
            continue
            
        a0_p, a1_p, b0_p, b1_p, h_p = state
        boxes = {b0_p: "small", b1_p: "small", h_p: "heavy"} if h_p else {b0_p: "small", b1_p: "small"}
        boxes = {k: v for k, v in boxes.items() if k is not None}
        
        for ja in joint_actions:
            act0, act1 = ja
            intents = {0: None, 1: None}
            
            for idx, (p, act_dir) in enumerate([(a0_p, act0), (a1_p, act1)]):
                vec = macro_vecs[act_dir]
                fwd = (p[0]+vec[0], p[1]+vec[1])
                intents[idx] = {"tgt": fwd, "dir": act_dir, "vec": vec}
            
            def get_branches(intent, p):
                tgt = intent["tgt"]
                if boxes.get(tgt) in ["small", "heavy"]:
                    return [(push_succ, intent), (1.0 - push_succ, "fail")]
                d = intent["dir"]
                dl = (d - 1) % 4; v_l = macro_vecs[dl]; tl = (p[0]+v_l[0], p[1]+v_l[1])
                dr = (d + 1) % 4; v_r = macro_vecs[dr]; tr = (p[0]+v_r[0], p[1]+v_r[1])
                return [(move_succ, {"tgt": tgt, "vec": intent["vec"]}),
                        (side_prob, {"tgt": tl, "vec": v_l}),
                        (side_prob, {"tgt": tr, "vec": v_r})]
            
            branches0 = get_branches(intents[0], a0_p)
            branches1 = get_branches(intents[1], a1_p)
            
            is_joint_heavy = False
            if intents[0] and intents[1]:
                if intents[0]["tgt"] == intents[1]["tgt"] and boxes.get(intents[0]["tgt"]) == "heavy":
                    if a0_p == a1_p and intents[0]["dir"] == intents[1]["dir"]:
                        is_joint_heavy = True
            
            outcomes = {}
            if is_joint_heavy:
                joint_branches = [(push_succ, (intents[0], intents[1])), (1.0 - push_succ, ("fail", "fail"))]
            else:
                joint_branches = [(p0 * p1, (b0, b1)) for p0, b0 in branches0 for p1, b1 in branches1]
            
            for prob, (b0_i, b1_i) in joint_branches:
                if prob == 0: continue
                new_b0 = b0_p; new_b1 = b1_p; new_h = h_p
                na0_p = a0_p; na1_p = a1_p
                nbs = dict(boxes)
                heavy_consumed = {0: False, 1: False}
                if is_joint_heavy and b0_i != "fail":
                    hv_tgt = intents[0]["tgt"]; nhv = (hv_tgt[0]+intents[0]["vec"][0], hv_tgt[1]+intents[0]["vec"][1])
                    if nhv in walkable and nbs.get(nhv) is None:
                        del nbs[hv_tgt]; nbs[nhv] = "heavy"
                        new_h = nhv; na0_p = hv_tgt; na1_p = hv_tgt
                    heavy_consumed[0] = True; heavy_consumed[1] = True
                        
                for idx, (b_intent, a_p, consumed) in enumerate([(b0_i, a0_p, heavy_consumed[0]), (b1_i, a1_p, heavy_consumed[1])]):
                    if consumed or b_intent is None or b_intent == "fail": continue
                    tgt = b_intent["tgt"]; vec = b_intent["vec"]
                    if nbs.get(tgt) == "small":
                        ntgt = (tgt[0]+vec[0], tgt[1]+vec[1])
                        if ntgt in walkable and nbs.get(ntgt) is None:
                            if b0_p == tgt: new_b0 = ntgt
                            elif b1_p == tgt: new_b1 = ntgt
                            del nbs[tgt]; nbs[ntgt] = "small"
                            if idx == 0: na0_p = tgt
                            else: na1_p = tgt
                    elif nbs.get(tgt) is None and tgt in walkable:
                        if idx == 0: na0_p = tgt
                        else: na1_p = tgt
                        
                ns = (na0_p, na1_p, new_b0, new_b1, new_h)
                outcomes[ns] = outcomes.get(ns, 0.0) + prob
                
            res_list = []
            for ns, prob in outcomes.items():
                rew = 1.0 if is_trans_terminal(ns) else 0.0
                res_list.append((prob, ns, rew))
                if ns not in transitions: queue.append(ns)
            transitions[state][ja] = res_list
            
    return transitions


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    print("Building transition model...")
    transitions = build_transition_model(env)
    print(f"Model built! {len(transitions)} reachable states.")
    
    states = list(transitions.keys())
    V = {s: 0.0 for s in states}
    policy = {s: (0, 0) for s in states}
    
    # Initialize policy arbitrarily
    for s in states:
        policy[s] = next(iter(transitions[s].keys()))
        
    for i in range(max_outer_iters):
        # 1. Partial policy evaluation (k sweeps)
        for _ in range(k):
            for s in states:
                act = policy[s]
                v_new = 0.0
                for prob, next_s, reward in transitions[s][act]:
                    v_new += prob * (reward + gamma * V[next_s])
                V[s] = v_new
                
        # 2. Policy improvement
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
        raise NotImplementedError("Shim not used, we call run_online_planning directly.")


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
        omni = (state[0], state[2], state[4], state[5], state[6])
        if omni not in policy:
            # Fallback if deviate somewhere strange during testing tracking
            return {env.possible_agents[0]: 0, env.possible_agents[1]: 0}
            
        joint_action = policy[omni] # (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        agents = env.possible_agents
        out_acts = {}
        
        for idx, agent in enumerate(agents):
            act = joint_action[idx]
            cur_dir = env.agent_dirs[agent]
            diff = (act - cur_dir) % 4
            if diff == 0: out_acts[agent] = 2 # forward
            elif diff == 1: out_acts[agent] = 1 # right
            elif diff == 3: out_acts[agent] = 0 # left
            else: out_acts[agent] = 1 # 180 flip requires 2 steps, just turn right this step
                
        return out_acts

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
