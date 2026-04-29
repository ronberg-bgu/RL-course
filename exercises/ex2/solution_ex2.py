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

import time
from collections import defaultdict

import numpy as np
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

# ---------------------------------------------------------------------------
# Map used in both parts (same as Assignment 1) — reduced to 6×6 for MPI tractability.
# ---------------------------------------------------------------------------
ASCII_MAP = [
    "WWWWWW",
    "W AA W",
    "W BB W",
    "W  C W",
    "WGG GW",
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

    # Rebuild missing box from agent-on-goal cells if grid scan came up short.
    if box0_pos is None or box1_pos is None:
        goal_set = set(getattr(env, "goal_positions", ()) or ())
        agent_cells = [a0_pos, a1_pos]
        candidates = [c for c in agent_cells if c in goal_set] or agent_cells
        if box0_pos is None:
            box0_pos = candidates[0]
        if box1_pos is None:
            box1_pos = candidates[-1]

    return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)


# Canonical state: no direction, sorted agents and boxes.
EAST, SOUTH, WEST, NORTH, STAY = 0, 1, 2, 3, 4
DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))   # MiniGrid order: E, S, W, N
JOINT_ACTIONS = [(d0, d1) for d0 in range(5) for d1 in range(5)]


def _extract_static(env):
    walls = set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                walls.add((x, y))
    return walls, frozenset(env.goal_positions)


def _initial_canonical(env):
    a_cells = sorted(env.agent_positions[a] for a in env.possible_agents)
    smalls, heavies = [], []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavies.append((x, y))
                else:
                    smalls.append((x, y))
    smalls.sort()
    return (a_cells[0], a_cells[1], smalls[0], smalls[1], heavies[0])


def _canonical(a0, a1, b0, b1, heavy):
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0
    return (a0, a1, b0, b1, heavy)


def _is_terminal(state, goals):
    _, _, b0, b1, heavy = state
    return b0 in goals and b1 in goals and heavy in goals


def _agent_outcome(pos, d, b0, b1, heavy, walls, p_move, p_push, other_pos):
    """Stochastic outcomes for one agent acting under heading d.
    Yields (prob, new_pos, new_b0, new_b1, new_heavy)."""
    if d == STAY:
        return [(1.0, pos, b0, b1, heavy)]

    dx, dy = DIR_VEC[d]
    fwd = (pos[0] + dx, pos[1] + dy)

    # Wall or heavy box ahead -> no-op (one agent can't push heavy alone).
    if fwd in walls or fwd == heavy:
        return [(1.0, pos, b0, b1, heavy)]

    # Small box ahead: push if the cell behind it is free.
    if fwd == b0 or fwd == b1:
        beyond = (fwd[0] + dx, fwd[1] + dy)
        if (beyond in walls or beyond == heavy or beyond == b0 or beyond == b1
                or beyond == other_pos):
            return [(1.0, pos, b0, b1, heavy)]
        if fwd == b0:
            nb0, nb1 = beyond, b1
        else:
            nb0, nb1 = b0, beyond
        if nb0 > nb1:
            nb0, nb1 = nb1, nb0
        return [
            (p_push,       fwd, nb0, nb1, heavy),
            (1.0 - p_push, pos, b0,  b1,  heavy),
        ]

    # Empty cell ahead: 0.8 intended, 0.1 each lateral. Blocked deviations stay.
    side = (1.0 - p_move) / 2.0
    out = []
    for prob, td in ((p_move, d), (side, (d - 1) % 4), (side, (d + 1) % 4)):
        tdx, tdy = DIR_VEC[td]
        target = (pos[0] + tdx, pos[1] + tdy)
        if target in walls or target == heavy or target == b0 or target == b1:
            out.append((prob, pos, b0, b1, heavy))
        else:
            out.append((prob, target, b0, b1, heavy))
    return out


def _simulate(state, joint_action, walls, p_move, p_push):
    """Returns list of (next_state, prob) for one joint action from one state."""
    a0, a1, b0, b1, heavy = state
    d0, d1 = joint_action

    # Joint heavy push: both agents on the same cell, same heading, heavy ahead.
    if d0 == d1 and d0 != STAY and a0 == a1:
        dx, dy = DIR_VEC[d0]
        fwd = (a0[0] + dx, a0[1] + dy)
        if fwd == heavy:
            beyond = (fwd[0] + dx, fwd[1] + dy)
            if beyond in walls or beyond == b0 or beyond == b1:
                return [(state, 1.0)]
            new_state = _canonical(fwd, fwd, b0, b1, beyond)
            return [(new_state, p_push), (state, 1.0 - p_push)]

    # Independent agents: process agent 0, then agent 1 sees the result.
    out = defaultdict(float)
    for p0, na0, nb00, nb01, nh0 in _agent_outcome(a0, d0, b0, b1, heavy, walls, p_move, p_push, other_pos=a1):
        for p1, na1, nb10, nb11, nh1 in _agent_outcome(a1, d1, nb00, nb01, nh0, walls, p_move, p_push, other_pos=na0):
            ns = _canonical(na0, na1, nb10, nb11, nh1)
            out[ns] += p0 * p1
    return list(out.items())


def _env_action(cur_dir, target_dir):
    """Translate an abstract heading into the env action (0=left, 1=right, 2=forward)."""
    if target_dir == STAY:
        return 0  # harmless rotation, no movement
    if cur_dir == target_dir:
        return 2
    diff = (target_dir - cur_dir) % 4
    return 1 if diff in (1, 2) else 0


class Policy:
    """policy[full_state] -> (env_action_0, env_action_1).

    Bridges the env's 7-tuple state (with directions, ordered by physical agent)
    and the abstract canonical 5-tuple state (no directions, sorted).
    """

    def __init__(self, abstract_pi, state_id):
        self._pi = abstract_pi          # list[int]: state_id -> joint-action id
        self._state_id = state_id       # dict: canonical state -> state_id

    def __getitem__(self, full_state):
        a0_pos, a0_dir, a1_pos, a1_dir, b0, b1, heavy = full_state

        # Canonicalize agents and remember which physical agent ended up first.
        if a0_pos <= a1_pos:
            ca0, ca1 = a0_pos, a1_pos
            slot_of_phys0, slot_of_phys1 = 0, 1
        else:
            ca0, ca1 = a1_pos, a0_pos
            slot_of_phys0, slot_of_phys1 = 1, 0

        # Canonicalize anonymous small boxes.
        cb0, cb1 = (b0, b1) if b0 <= b1 else (b1, b0)

        sid = self._state_id.get((ca0, ca1, cb0, cb1, heavy))
        if sid is None:
            return (2, 2)  # both forward — escape unrecognized state
        slot0_heading, slot1_heading = JOINT_ACTIONS[self._pi[sid]]
        slots = (slot0_heading, slot1_heading)

        return (
            _env_action(a0_dir, slots[slot_of_phys0]),
            _env_action(a1_dir, slots[slot_of_phys1]),
        )


def build_transition_model(env):
    """BFS the reachable canonical states and record P(s' | s, a) as a plain dict.

    Returns (state_id, transitions, terminals) where:
      state_id    : dict canonical 5-tuple state -> integer id
      transitions : dict (sid, action_id) -> list of (next_sid, prob)
      terminals   : set of state ids satisfying _all_boxes_on_goals.

    BFS rather than full enumeration: walls and the heavy box make many joint
    positions unreachable, so we only allocate states the policy can visit.
    """
    if not getattr(env, "goal_positions", None):
        env.reset()

    walls, goals = _extract_static(env)
    p_move, p_push = env.move_success_prob, env.push_success_prob

    s0 = _initial_canonical(env)
    state_id = {s0: 0}
    state_list = [s0]
    transitions = {}
    terminals = set()

    queue = [0]
    t0 = time.time()
    while queue:
        sid = queue.pop()
        state = state_list[sid]
        if _is_terminal(state, goals):
            terminals.add(sid)
            continue
        for ai, joint_action in enumerate(JOINT_ACTIONS):
            outcomes = []
            for next_state, prob in _simulate(state, joint_action, walls, p_move, p_push):
                if next_state not in state_id:
                    state_id[next_state] = len(state_list)
                    state_list.append(next_state)
                    queue.append(state_id[next_state])
                outcomes.append((state_id[next_state], prob))
            transitions[(sid, ai)] = outcomes
        if len(state_list) % 10000 == 0 and len(state_list) > 0:
            print(f"  BFS: {len(state_list):,} states ({time.time() - t0:.1f}s)")

    n_trans = sum(len(v) for v in transitions.values())
    print(f"  reachable: {len(state_list):,} states "
          f"({len(terminals):,} terminal), {n_trans:,} transitions, "
          f"build {time.time() - t0:.1f}s")

    return state_id, transitions, terminals


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """Modified Policy Iteration.

    Each outer iteration:
      1. Partial policy evaluation — up to k Bellman sweeps under π,
         stopping early once max|V_new − V_old| < theta.
              V[s] ← −1 + γ · Σ_{s'} P(s' | s, π(s)) · V[s']
      2. Policy improvement — greedy w.r.t. the freshly evaluated V.
    Outer terminates on policy stability (policy-improvement theorem).
    Reward is −1 per non-terminal step, 0 at terminals; V[s] is the
    (negative) expected discounted steps from s to a goal.
    """
    print("Building transition model…")
    state_id, transitions, terminals = build_transition_model(env)
    N = len(state_id)
    A = len(JOINT_ACTIONS)
    print(f"Running MPI: gamma={gamma}, k={k}, theta={theta}, |A|={A}, N={N:,}")

    # Heuristic V init: −Σ Manhattan(box, nearest goal). Speeds convergence without changing the optimum.
    goals = list(env.goal_positions)
    def _box_dist(p):
        return min(abs(p[0]-gx) + abs(p[1]-gy) for gx, gy in goals)
    id_to_state = [None] * N
    for s_tuple, sid in state_id.items():
        id_to_state[sid] = s_tuple
    V = [0.0] * N
    for sid in range(N):
        if sid in terminals:
            continue
        _, _, b0, b1, heavy = id_to_state[sid]
        V[sid] = -float(_box_dist(b0) + _box_dist(b1) + _box_dist(heavy))
    policy = [0] * N

    t0 = time.time()
    for outer in range(max_outer_iters):
        V_outer_start = list(V)

        # 1. Policy evaluation: up to k sweeps, early-stop on theta.
        for _ in range(k):
            delta = 0.0
            for s in range(N):
                if s in terminals:
                    V[s] = 0.0
                    continue
                a = policy[s]
                v_new = -1.0 + gamma * sum(p * V[ns] for ns, p in transitions[(s, a)])
                if abs(v_new - V[s]) > delta:
                    delta = abs(v_new - V[s])
                V[s] = v_new
            if delta < theta:
                break

        # 2. Policy improvement: greedy with sticky tiebreak.
        new_policy = [0] * N
        for s in range(N):
            if s in terminals:
                continue
            cur = policy[s]
            best_a = cur
            best_q = -1.0 + gamma * sum(p * V[ns] for ns, p in transitions[(s, cur)])
            for a in range(A):
                if a == cur:
                    continue
                q = -1.0 + gamma * sum(p * V[ns] for ns, p in transitions[(s, a)])
                if q > best_q + 1e-9:
                    best_q, best_a = q, a
            new_policy[s] = best_a

        n_changed = sum(1 for s in range(N) if new_policy[s] != policy[s])
        outer_delta = max(abs(V[s] - V_outer_start[s]) for s in range(N))
        print(f"  outer {outer + 1}: {n_changed:,} policy changes, "
              f"max|ΔV|={outer_delta:.6f} ({time.time() - t0:.1f}s)")
        policy = new_policy
        if outer_delta < theta:
            break

    return Policy(policy, state_id), V


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
