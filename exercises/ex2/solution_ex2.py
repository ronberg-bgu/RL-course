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
from collections import deque, defaultdict
import itertools
import random

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# MiniGrid directions: 0=right, 1=down, 2=left, 3=up
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
JOINT_ACTIONS = [
    (a0, a1) for a0 in range(3) for a1 in range(3)
]  # 9 joint actions: (0..2) x (0..2)


# ===========================================================================
# Part 1 — Online Planning
# ===========================================================================

def _execute_pddl_action(env, pddl_action):
    """
    Convert a single PDDL action into low-level env steps (rotations + forward).

    Returns
    -------
    obs, rewards, terms, truncs : env step outputs from the last sub-step
    num_steps : int  — number of env.step() calls made
    done : bool — True if any agent terminated or truncated
    """
    agent_targets = extract_target_pos(pddl_action)
    if not agent_targets:
        return None, None, None, None, 0, False

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

    num_steps = 0
    done = False
    obs = rewards = terms = truncs = None

    while any(len(q) > 0 for q in action_queues.values()):
        step_actions = {}
        for a in agents_in_action:
            if action_queues[a]:
                act = action_queues[a].pop(0)
                if act is not None:
                    step_actions[a] = act

        obs, rewards, terms, truncs, _ = env.step(step_actions)
        num_steps += 1

        if any(terms.values()) or any(truncs.values()):
            done = True
            break

    return obs, rewards, terms, truncs, num_steps, done


def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Execute one episode using online planning:
      replan from the current state → execute only the first PDDL action → repeat.

    When a stochastic push fails (state unchanged), we retry the same action
    up to MAX_RETRIES times before wasting time on an expensive replan call.

    Returns
    -------
    int
        Number of *env* steps taken (counting each rotate/forward individually).
    """
    MAX_RETRIES = 5  # retry same action before replanning

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

        # ── 3. Execute the first PDDL action (with retries) ──────────
        pddl_action = plan.actions[0]
        prev_state = get_state(env)

        for _retry in range(MAX_RETRIES):
            obs, rewards, terms, truncs, steps, done = _execute_pddl_action(
                env, pddl_action
            )
            total_env_steps += steps

            if done or steps == 0:
                break

            # If state changed, the action succeeded — move on to replan
            if get_state(env) != prev_state:
                break
            # Otherwise the push failed stochastically — retry same action

        # If still stuck after all retries, random perturbation
        if not done and get_state(env) == prev_state:
            random_actions = {a: random.choice([0, 1, 2]) for a in env.agents}
            obs, rewards, terms, truncs, _ = env.step(random_actions)
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True

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


# ---------------------------------------------------------------------------
# Transition model helpers  (pure functions, easily testable)
# ---------------------------------------------------------------------------

def extract_walls(env):
    """Return a frozenset of (x, y) wall positions from a live environment."""
    walls = set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                walls.add((x, y))
    return frozenset(walls)


def is_terminal(state, goals):
    """Check whether all boxes are on goal positions."""
    boxes = [b for b in [state[4], state[5], state[6]] if b is not None]
    if len(goals) == 0 or len(boxes) == 0:
        return False
    return all(b in goals for b in boxes)


def is_blocked(pos, walls, box_positions):
    """Check if a position is occupied by a wall or any box."""
    if pos in walls:
        return True
    if pos in box_positions:
        return True
    return False


def compute_new_directions(a0_d, a1_d, a0_act, a1_act):
    """Compute new agent directions after rotation actions."""
    new_d0 = a0_d
    new_d1 = a1_d
    if a0_act == 0:
        new_d0 = (a0_d - 1) % 4
    elif a0_act == 1:
        new_d0 = (a0_d + 1) % 4
    if a1_act == 0:
        new_d1 = (a1_d - 1) % 4
    elif a1_act == 1:
        new_d1 = (a1_d + 1) % 4
    return new_d0, new_d1


def compute_forward_intents(a0_p, a0_d, a0_act, a1_p, a1_d, a1_act):
    """
    For agents performing action==2 (forward), compute their forward intent.
    Returns a dict  agent_idx -> {pos, target, dir, vec}.
    """
    intents = {}
    for idx, pos, d, act in [(0, a0_p, a0_d, a0_act), (1, a1_p, a1_d, a1_act)]:
        if act == 2:
            vec = DIR_TO_VEC[d]
            fwd_pos = (pos[0] + vec[0], pos[1] + vec[1])
            intents[idx] = {"pos": pos, "target": fwd_pos, "dir": d, "vec": vec}
    return intents


def resolve_heavy_push(intents, a0_p, a1_p, h_p, b0_p, b1_p, walls):
    """
    Determine if a valid heavy push occurs and consume the agents' intents.

    Returns
    -------
    heavy_push_dest : tuple or None
        The destination of the heavy box if push preconditions are met.
    consumed : set
        Indices of agents consumed by the heavy push attempt.
    intents : dict
        Updated intents with consumed agents removed.
    """
    if h_p is None:
        return None, set(), intents

    heavy_pushers = []
    for idx, intent in list(intents.items()):
        if intent["target"] == h_p:
            heavy_pushers.append((idx, intent))

    consumed = set()
    heavy_push_dest = None

    if len(heavy_pushers) >= 2:
        origins = set(intent["pos"] for idx, intent in heavy_pushers)
        dirs = set(intent["dir"] for idx, intent in heavy_pushers)

        if len(dirs) == 1 and len(origins) == 1:
            p_dir = next(iter(dirs))
            vec = DIR_TO_VEC[p_dir]
            nx, ny = h_p[0] + vec[0], h_p[1] + vec[1]
            box_positions = {b0_p, b1_p} - {None}
            if not is_blocked((nx, ny), walls, box_positions):
                heavy_push_dest = (nx, ny)

        # Consume all pushers regardless of success (matches env behavior)
        for idx, _ in heavy_pushers:
            consumed.add(idx)
            intents.pop(idx, None)

    return heavy_push_dest, consumed, intents


def resolve_small_pushes(intents, b0_p, b1_p, h_p, walls):
    """
    Identify which remaining agents are pushing a small box.

    Returns
    -------
    small_push_events : dict
        agent_idx -> (box_idx, new_box_pos)  for valid pushes.
    """
    small_push_events = {}
    box_positions = {b0_p, b1_p, h_p} - {None}

    for idx, intent in intents.items():
        t_p = intent["target"]
        vec = intent["vec"]

        box_idx = None
        if t_p == b0_p:
            box_idx = 0
        elif t_p == b1_p:
            box_idx = 1

        if box_idx is not None:
            nx, ny = t_p[0] + vec[0], t_p[1] + vec[1]
            # Check if box destination is clear (no wall, no other box)
            if not is_blocked((nx, ny), walls, box_positions):
                small_push_events[idx] = (box_idx, (nx, ny))

    return small_push_events


def resolve_moves(intents, b0_p, b1_p, h_p, walls, move_prob, small_push_events):
    """
    For remaining forward-moving agents (not pushing), compute stochastic
    move outcomes.

    Returns
    -------
    move_events : dict
        agent_idx -> list of (probability, resulting_position)
    """
    move_events = {}
    box_positions = {b0_p, b1_p, h_p} - {None}

    for idx, intent in intents.items():
        if idx in small_push_events:
            continue  # already handled as a push

        t_p = intent["target"]
        d = intent["dir"]
        agent_pos = intent["pos"]

        # Forward cell must be clear for a move (not a wall, not a box)
        if is_blocked(t_p, walls, box_positions):
            continue  # no-op, agent stays (not even stochastic)

        # Precondition met: compute stochastic outcomes
        side_prob = (1.0 - move_prob) / 2.0
        events = []

        # Straight
        events.append((move_prob, t_p))
        # Left deviation
        ldir = (d - 1) % 4
        lp = (agent_pos[0] + DIR_TO_VEC[ldir][0], agent_pos[1] + DIR_TO_VEC[ldir][1])
        events.append((side_prob, lp))
        # Right deviation
        rdir = (d + 1) % 4
        rp = (agent_pos[0] + DIR_TO_VEC[rdir][0], agent_pos[1] + DIR_TO_VEC[rdir][1])
        events.append((side_prob, rp))

        # If deviated cell is blocked, agent stays put
        clean_events = []
        for p, pos in events:
            if is_blocked(pos, walls, box_positions):
                clean_events.append((p, agent_pos))
            else:
                clean_events.append((p, pos))

        move_events[idx] = clean_events

    return move_events


def combine_outcomes(
    a0_p, a1_p, b0_p, b1_p, h_p, new_d0, new_d1,
    heavy_push_dest, consumed,
    small_push_events, move_events,
    push_prob, goals,
):
    """
    Combine independent stochastic events into a list of (prob, next_state, reward).
    """
    independent_outcomes = []

    # Heavy push outcome
    if heavy_push_dest is not None:
        independent_outcomes.append([
            {"prob": push_prob, "h_p": heavy_push_dest, "a0_p": h_p, "a1_p": h_p},
            {"prob": 1.0 - push_prob, "h_p": h_p, "a0_p": a0_p, "a1_p": a1_p},
        ])

    # Per-agent outcomes
    for idx in [0, 1]:
        if idx in consumed:
            continue

        agent_pos = a0_p if idx == 0 else a1_p
        key_a = f"a{idx}_p"

        if idx in small_push_events:
            box_idx, new_box_pos = small_push_events[idx]
            old_box_pos = b0_p if box_idx == 0 else b1_p
            key_b = f"b{box_idx}_p"
            independent_outcomes.append([
                {"prob": push_prob, key_a: old_box_pos, key_b: new_box_pos},
                {"prob": 1.0 - push_prob, key_a: agent_pos, key_b: old_box_pos},
            ])
        elif idx in move_events:
            evs = [{"prob": p, key_a: pos} for p, pos in move_events[idx]]
            independent_outcomes.append(evs)
        else:
            independent_outcomes.append([{"prob": 1.0, key_a: agent_pos}])

    # Cartesian product of all independent outcomes
    state_probs = defaultdict(float)
    state_rewards = {}
    for combination in itertools.product(*independent_outcomes):
        prob = 1.0
        updates = {}
        for outcome in combination:
            prob *= outcome["prob"]
            for k, v in outcome.items():
                if k != "prob":
                    updates[k] = v
        if prob <= 0:
            continue

        na0_p = updates.get("a0_p", a0_p)
        na1_p = updates.get("a1_p", a1_p)
        nb0_p = updates.get("b0_p", b0_p)
        nb1_p = updates.get("b1_p", b1_p)
        nh_p = updates.get("h_p", h_p)
        ns = (na0_p, new_d0, na1_p, new_d1, nb0_p, nb1_p, nh_p)
        reward = 1.0 if is_terminal(ns, goals) else 0.0
        state_probs[ns] += prob
        state_rewards[ns] = reward

    return [(p, s, state_rewards[s]) for s, p in state_probs.items()]


def step_state(state, joint_action, walls, goals, move_prob, push_prob):
    """
    Compute all stochastic outcomes of applying *joint_action* in *state*.

    Returns
    -------
    list of (probability, next_state, reward)
    """
    a0_p, a0_d, a1_p, a1_d, b0_p, b1_p, h_p = state
    a0_act, a1_act = joint_action

    # 1. Directions
    new_d0, new_d1 = compute_new_directions(a0_d, a1_d, a0_act, a1_act)

    # 2. Forward intents
    intents = compute_forward_intents(a0_p, a0_d, a0_act, a1_p, a1_d, a1_act)

    # 3. Heavy push
    heavy_push_dest, consumed, intents = resolve_heavy_push(
        intents, a0_p, a1_p, h_p, b0_p, b1_p, walls
    )

    # 4. Small pushes
    small_push_events = resolve_small_pushes(intents, b0_p, b1_p, h_p, walls)

    # 5. Moves
    move_events = resolve_moves(
        intents, b0_p, b1_p, h_p, walls, move_prob, small_push_events
    )

    # 6. Combine
    return combine_outcomes(
        a0_p, a1_p, b0_p, b1_p, h_p, new_d0, new_d1,
        heavy_push_dest, consumed,
        small_push_events, move_events,
        push_prob, goals,
    )


def build_transition_model(env):
    """
    Build the full MDP transition model analytically via BFS over reachable states.

    Returns
    -------
    transitions : dict
        transitions[state][joint_action] = [(prob, next_state, reward), ...]
    """
    goals = frozenset(env.goal_positions)
    walls = extract_walls(env)
    move_prob = env.move_success_prob
    push_prob = env.push_success_prob

    start_state = get_state(env)
    print("Building transition model (BFS reachable states)...")
    queue = deque([start_state])
    visited = {start_state}
    transitions = {}

    count = 0
    while queue:
        s = queue.popleft()
        transitions[s] = {}
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} states...")

        if is_terminal(s, goals):
            for a in JOINT_ACTIONS:
                transitions[s][a] = [(1.0, s, 0.0)]
            continue

        for a in JOINT_ACTIONS:
            next_states = step_state(s, a, walls, goals, move_prob, push_prob)
            transitions[s][a] = next_states
            for p, ns, r in next_states:
                if ns not in visited:
                    visited.add(ns)
                    queue.append(ns)

    print(f"Model built. {len(transitions)} reachable states.")
    return transitions


# ---------------------------------------------------------------------------
# MPI helpers
# ---------------------------------------------------------------------------

def policy_evaluation_step(V, policy, transitions, states, gamma):
    """One synchronous sweep of policy evaluation. Returns new V."""
    V_new = {}
    for s in states:
        a = policy[s]
        v = 0.0
        for prob, ns, reward in transitions[s][a]:
            v += prob * (reward + gamma * V[ns])
        V_new[s] = v
    return V_new


def policy_improvement(V, transitions, states, actions, gamma):
    """
    Greedy policy improvement.

    Returns
    -------
    new_policy : dict
    new_V : dict  (updated with best action values)
    changed : bool  (True if any action changed)
    max_v_change : float
    """
    new_policy = {}
    new_V = {}
    changed = False
    max_v_change = 0.0

    for s in states:
        best_a = actions[0]
        best_v = float("-inf")

        for a in actions:
            v = 0.0
            for prob, ns, reward in transitions[s][a]:
                v += prob * (reward + gamma * V[ns])
            if v > best_v:
                best_v = v
                best_a = a

        new_policy[s] = best_a
        max_v_change = max(max_v_change, abs(V[s] - best_v))
        new_V[s] = best_v
        if s not in new_policy or new_policy[s] != best_a:
            pass  # already set
        # track change relative to old implicit policy
        changed = changed or (V.get(s, 0.0) != best_v)

    return new_policy, new_V, changed, max_v_change


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration.

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
    transitions = build_transition_model(env)
    states = list(transitions.keys())

    V = {s: 0.0 for s in states}
    policy = {s: (0, 0) for s in states}

    print("Starting Modified Policy Iteration...")
    for i in range(max_outer_iters):
        # Policy Evaluation (k steps)
        for _ in range(k):
            V = policy_evaluation_step(V, policy, transitions, states, gamma)

        # Policy Improvement
        old_policy = policy.copy()
        policy_stable = True
        max_v_change = 0.0

        for s in states:
            old_a = old_policy[s]
            best_a = old_a
            best_v = float("-inf")

            for a in JOINT_ACTIONS:
                v = 0.0
                for prob, ns, reward in transitions[s][a]:
                    v += prob * (reward + gamma * V[ns])
                if v > best_v:
                    best_v = v
                    best_a = a

            policy[s] = best_a
            max_v_change = max(max_v_change, abs(V[s] - best_v))
            V[s] = best_v
            if old_a != best_a:
                policy_stable = False

        print(f"Iteration {i+1}: max Value change = {max_v_change:.6f}")

        if policy_stable and max_v_change < theta:
            print(f"MPI converged after {i+1} iterations.")
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
