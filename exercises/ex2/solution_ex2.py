"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""
import itertools
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
    "WWWWW",
    "WAA W",
    "WBCBW",
    "WGGGW",
    "WWWWW",
]

ASCII_MAP = [
    "WWWWW",
    "WAA W",
    "WBCBW",
    "W   W",
    "W   W",
    "WGGGW",
    "WWWWW",
]

# ASCII_MAP = [
#     "WWWWWWWW",
#     "W  AA  W",
#     "W B C  W",
#     "W      W",
#     "W   B  W",
#     "W G G GW",
#     "WWWWWWWW",
# ]


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
    # a0_dir = env.agent_dirs[agents[0]] # even though that in the env dir is relevant in our problem this is ignored
    a1_pos = env.agent_positions[agents[1]]
    # a1_dir = env.agent_dirs[agents[1]]

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

    # return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)
    return (a0_pos, a1_pos, box0_pos, box1_pos, heavy_pos)


def build_transition_model(env):
    print("🔍 Building Transition Model...")
    transitions = {}

    env.reset()

    w = env.width
    h = env.height

    # ── 1. Map the static environment ──────────────────────────────
    walls = set()
    goals = set()
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is not None:
                if cell.type == "wall":
                    walls.add((x, y))
                elif cell.type == "goal":
                    goals.add((x, y))

    def is_terminal(state):
        b0, b1, heavy = state[2], state[3], state[4]
        boxes = [b for b in [b0, b1, heavy] if b is not None]
        return len(boxes) > 0 and all(b in goals for b in boxes)

    def is_open(p):
        return 0 <= p[0] < w and 0 <= p[1] < h and p not in walls

    DIR_VECS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    # ── 2. PHYSICS HELPER: Individual Math ─────────────────────────
    def get_single_agent_outcomes(pos, act, small_boxes, all_boxes, heavy_pos):
        """Calculates intended moves and slips for a single agent."""
        dx, dy = DIR_VECS[act]
        fwd_pos = (pos[0] + dx, pos[1] + dy)

        # Trying to push a heavy box individually always fails
        if fwd_pos == heavy_pos:
            return [(1.0, pos, None, None)]

        # Trying to push a small box
        if fwd_pos in small_boxes:
            fwd_fwd = (fwd_pos[0] + dx, fwd_pos[1] + dy)
            if is_open(fwd_fwd) and fwd_fwd not in all_boxes:
                return [
                    (env.push_success_prob, fwd_pos, fwd_pos, fwd_fwd),
                    (1.0 - env.push_success_prob, pos, None, None)
                ]
            else:
                return [(1.0, pos, None, None)]

        # Moving (Intended + Slipping)
        outcomes = []
        if is_open(fwd_pos) and fwd_pos not in all_boxes:
            outcomes.append((env.move_success_prob, fwd_pos, None, None))
        else:
            outcomes.append((env.move_success_prob, pos, None, None))

        slip_prob = (1.0 - env.move_success_prob) / 2.0
        for slip_act in [(act - 1) % 4, (act + 1) % 4]:
            sdx, sdy = DIR_VECS[slip_act]
            s_pos = (pos[0] + sdx, pos[1] + sdy)
            if is_open(s_pos) and s_pos not in all_boxes:
                outcomes.append((slip_prob, s_pos, None, None))
            else:
                outcomes.append((slip_prob, pos, None, None))
        return outcomes

    # ── 3. PHYSICS HELPER: Joint Action Multiverse ─────────────────
    def calculate_outcomes_list(state, j_act):
        """Returns the final [(prob, next_state, reward), ...] for a joint action."""
        a0_pos, a1_pos, b0_pos, b1_pos, heavy_pos = state
        small_boxes = [b for b in [b0_pos, b1_pos] if b is not None]
        all_boxes = small_boxes + ([heavy_pos] if heavy_pos else [])
        a0_act, a1_act = j_act

        outcomes_dict = {}

        # --- THE HEAVY BOX EXCEPTION ---
        if heavy_pos and a0_pos == a1_pos and a0_act == a1_act:
            dx, dy = DIR_VECS[a0_act]
            fwd_pos = (a0_pos[0] + dx, a0_pos[1] + dy)

            if fwd_pos == heavy_pos:
                fwd_fwd = (heavy_pos[0] + dx, heavy_pos[1] + dy)
                if is_open(fwd_fwd) and fwd_fwd not in small_boxes:
                    success_s = (fwd_pos, fwd_pos, b0_pos, b1_pos, fwd_fwd)
                    outcomes_dict[(success_s, -1.0)] = env.push_success_prob
                    outcomes_dict[(state, -1.0)] = 1.0 - env.push_success_prob
                    return [(p, ns, r) for (ns, r), p in outcomes_dict.items()]

        # --- NORMAL INDEPENDENT RESOLUTION ---
        a0_outcomes = get_single_agent_outcomes(a0_pos, a0_act, small_boxes, all_boxes, heavy_pos)
        a1_outcomes = get_single_agent_outcomes(a1_pos, a1_act, small_boxes, all_boxes, heavy_pos)

        for combo in itertools.product(a0_outcomes, a1_outcomes):
            (p0, n0_pos, b0_push, b0_new) = combo[0]
            (p1, n1_pos, b1_push, b1_new) = combo[1]

            joint_prob = p0 * p1

            # --- START OF FIX ---
            # 1. Collision check: If they both push a box, ensure they don't break physics
            cancel_pushes = False
            if b0_push and b1_push:
                if b0_new == b1_new: cancel_pushes = True # Pushed into the same empty square
                if b0_push == b1_push: cancel_pushes = True # Both pushed the exact same small box

            if cancel_pushes:
                b0_push, b1_push = None, None
                n0_pos, n1_pos = a0_pos, a1_pos # Cancel moves and bounce agents back

            # 2. Reconstruct the small boxes safely by tracking coordinates
            final_small = []
            for b in small_boxes:
                if b == b0_push: final_small.append(b0_new)
                elif b == b1_push: final_small.append(b1_new)
                else: final_small.append(b)

            # 3. CRITICAL: Sort the boxes so they don't swap slots and multiply states!
            final_small.sort()
            new_b0 = final_small[0] if len(final_small) > 0 else None
            new_b1 = final_small[1] if len(final_small) > 1 else None
            # --- END OF FIX ---

            next_s = (n0_pos, n1_pos, new_b0, new_b1, heavy_pos)
            key = (next_s, -1.0)
            outcomes_dict[key] = outcomes_dict.get(key, 0.0) + joint_prob

        return [(p, ns, r) for (ns, r), p in outcomes_dict.items()]

    # ── 4. Main BFS Execution Loop ─────────────────────────────────
    env.reset()
    start_state = get_state(env)
    queue = [start_state]
    visited = {start_state}

    actions = [0, 1, 2, 3]  # East, South, West, North
    all_joint_actions = list(itertools.product(actions, repeat=2))
    states_processed = 0

    while queue:
        state = queue.pop(0)
        transitions[state] = {}
        states_processed += 1

        if states_processed % 500 == 0:
            print(f"  ... processed {states_processed} reachable states")

        if is_terminal(state):
            for ja in all_joint_actions:
                transitions[state][ja] = [(1.0, state, 0.0)]
            continue

        # Check every joint action and cleanly save the outcomes!
        for j_act in all_joint_actions:
            final_outcomes = calculate_outcomes_list(state, j_act)
            transitions[state][j_act] = final_outcomes

            # Add newly discovered states to the queue
            for prob, ns, reward in final_outcomes:
                if ns not in visited:
                    visited.add(ns)
                    queue.append(ns)

    print(f"✅ Model Complete! Mapped {len(transitions)} reachable states.")
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
    print("\n🧠 Starting Modified Policy Iteration...")

    transitions = build_transition_model(env)
    states = list(transitions.keys())

    # 1: Initialize v_0 (arbitrarily)
    V = {s: 0.0 for s in states}
    policy = {}

    # 2: repeat
    for i in range(max_outer_iters):
        # Store v_k to check convergence at the end of the loop
        V_old_outer = V.copy()

        # 3: Policy Improvement (pi_{k+1})
        for s in states:
            best_action = None
            best_value = -float('inf')

            for action, outcomes in transitions[s].items():
                q_s_a = 0.0
                for prob, next_s, reward in outcomes:
                    q_s_a += prob * (reward + gamma * V[next_s])

                if q_s_a > best_value:
                    best_value = q_s_a
                    best_action = action

            policy[s] = best_action

        # 4, 5, 6, 7: Partial Evaluation (k sweeps)
        for _ in range(k):
            new_V = V.copy()
            for s in states:
                action = policy.get(s)
                if action is None: continue

                v_s = 0.0
                for prob, next_s, reward in transitions[s][action]:
                    v_s += prob * (reward + gamma * V[next_s])

                new_V[s] = v_s
            # Update v_{k, j}
            V = new_V

        # 8 & 9: Update and check convergence ||v_{k+1} - v_k|| < epsilon
        max_diff = max([abs(V[s] - V_old_outer[s]) for s in states])

        print(f"  Outer Iteration {i + 1} complete. Max Value Diff: {max_diff:.6f}")

        if max_diff < theta:
            print(f"🎯 MPI Converged to optimal value after {i + 1} iterations!")
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

    # # ── Part 1: Online Planning ──────────────────────────────────────────────
    # print("=" * 60)
    # print("Part 1 — Online Planning (classical planner on stochastic env)")
    # print("=" * 60)
    #
    # # Wrap run_online_planning as a policy function for the evaluator
    # def online_planning_policy(env, obs):
    #     """
    #     This wrapper runs one COMPLETE episode internally and is only a shim
    #     for the evaluator.  evaluate_policy will reset the env before each
    #     call, so we hand control back immediately with a do-nothing action
    #     after the first step — the real logic is inside run_online_planning.
    #
    #     NOTE: because run_online_planning drives the env loop itself, you
    #     should call it directly (see the manual loop below) for the 100-run
    #     evaluation; or adapt the evaluate_policy call to suit your design.
    #     """
    #     raise NotImplementedError(
    #         "Adapt this shim or call run_online_planning directly in a loop."
    #     )
    #
    # # Direct evaluation loop for online planning
    # online_steps = []
    # for i in range(1):
    #     env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    #     steps = run_online_planning(env_ep)
    #     online_steps.append(steps)
    #     if (i + 1) % 10 == 0:
    #         print(f"  run {i+1}/100 — steps so far: {steps}")
    #
    # mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    # print(f"\nOnline Planning  →  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 — Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)


    def mpi_policy_fn(env, obs):
        """Convert current env state to a joint action using the MPI policy."""
        state = get_state(env)

        # Safety fallback
        if state not in policy:
            agents = env.possible_agents
            return {agents[0]: 2, agents[1]: 2}

        joint_action = policy[state]
        agents = env.possible_agents
        sim_actions = {}

        for i, agent in enumerate(agents):
            target_dir = joint_action[i]
            current_dir = env.agent_dirs[agent]

            # Scenario 1: Facing the correct way
            if current_dir == target_dir:
                sim_actions[agent] = 2  # Command 2: Move Forward

            # Scenario 2: Target is 90 degrees to the right
            elif (current_dir + 1) % 4 == target_dir:
                sim_actions[agent] = 1  # Command 1: Turn Right

            # Scenario 3: Target is Left or directly Behind
            else:
                sim_actions[agent] = 0  # Command 0: Turn Left

        return sim_actions

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
