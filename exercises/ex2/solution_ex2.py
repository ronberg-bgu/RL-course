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
from collections import deque
import random
import logging
import time
try:
    from unified_planning.shortcuts import get_environment  # type: ignore
    get_environment().credits_stream = None  # silence planner credits
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging setup — writes to solution_ex2.log alongside this script
# ---------------------------------------------------------------------------
_log_path = os.path.join(os.path.dirname(__file__), "solution_ex2.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w"),  # overwrite each run
        logging.StreamHandler(),                     # also print to console
    ],
)
logger = logging.getLogger("solution_ex2")

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

# ---------------------------------------------------------------------------
# Map used in both parts (same as Assignment 1)
# ---------------------------------------------------------------------------
ASCII_MAP = [
    "WWWWWWW",
    "WA  A W",
    "W BB CW",
    "W     W",
    "WGG  GW",
    "WWWWWWW",
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

    for replan_idx in range(max_replans):
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
            # Agent isn't where the plan expects (stochastic deviation) — replan
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
# State representation (direction-free, following Omer's approach)
# ---------------------------------------------------------------------------
# A state is a tuple:
#   (agent0_pos, agent1_pos, box0_pos, box1_pos, heavy_pos)
#
# where positions are (col, row) tuples.
#
# Agent directions are excluded: rotations are deterministic, carry no reward,
# and can always be performed for free before any move/push. Therefore the
# optimal value of a state is independent of which direction agents currently
# face, reducing the state space significantly.

# ---------------------------------------------------------------------------
# MPI action space (abstract direction-based, one agent acts per step)
# ---------------------------------------------------------------------------
# MiniGrid direction convention: 0=right, 1=down, 2=left, 3=up
# Grid coordinates: (x, y) where (0,0) is top-left, x=col, y=row

NOOP  = 0
NORTH = 1   # MiniGrid dir=3, vec=(0,-1)
EAST  = 2   # MiniGrid dir=0, vec=(+1, 0)
SOUTH = 3   # MiniGrid dir=1, vec=(0,+1)
WEST  = 4   # MiniGrid dir=2, vec=(-1, 0)

ACTION_TO_DIR  = {NORTH: 3, EAST: 0, SOUTH: 1, WEST: 2}
MPI_DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIRS = [NORTH, EAST, SOUTH, WEST]

# 13 joint actions: sequential moves + coordinated heavy push
JOINT_ACTIONS = (
    [(d, NOOP) for d in DIRS] +   # only agent 0 acts
    [(NOOP, d) for d in DIRS] +   # only agent 1 acts
    [(NOOP, NOOP)]             +  # both idle
    [(d, d)    for d in DIRS]     # coordinated heavy-box push
)


def get_state(env) -> tuple:
    """Extract the current state tuple from a live environment."""
    agents = env.possible_agents
    a0_pos = env.agent_positions[agents[0]]
    a1_pos = env.agent_positions[agents[1]]

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

    box0_pos  = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos  = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos = heavy_boxes[0] if heavy_boxes else None

    return (a0_pos, a1_pos, box0_pos, box1_pos, heavy_pos)


# ---------------------------------------------------------------------------
# Transition model (direction-free, sequential agent execution)
# ---------------------------------------------------------------------------

def get_stochastic_outcomes(env, state, joint_action):
    """
    Return the stochastic outcomes of applying *joint_action* in *state*.

    Parameters
    ----------
    env          : StochasticMultiAgentBoxPushEnv
    state        : tuple  — as returned by get_state()
    joint_action : tuple  — (action_agent0, action_agent1) from JOINT_ACTIONS

    Returns
    -------
    list of (probability, next_state, reward) triples
    """
    a0_pos, a1_pos, box0_pos, box1_pos, heavy_pos = state
    a0_act, a1_act = joint_action

    walls  = frozenset(
        (x, y)
        for y, row in enumerate(env.ascii_map)
        for x, ch in enumerate(row)
        if ch == 'W'
    )
    goals    = frozenset(env.goal_positions)
    move_p   = env.move_success_prob
    push_p   = env.push_success_prob
    side_p   = (1.0 - move_p) / 2.0

    def blocked(pos, sboxes, hpos):
        """True if pos is a wall, small box, or heavy box."""
        return pos in walls or pos in sboxes or pos == hpos

    def reward(sb0, sb1, hp):
        box_set = {p for p in (sb0, sb1, hp) if p is not None}
        return 1.0 if goals <= box_set else 0.0

    def single_outcomes(agent_pos, action, sb0, sb1, hp):
        """Outcomes for one agent acting; returns [(prob, new_pos, sb0, sb1, hp)]."""
        if action == NOOP:
            return [(1.0, agent_pos, sb0, sb1, hp)]

        sboxes = frozenset(p for p in (sb0, sb1) if p is not None)
        mdir   = ACTION_TO_DIR[action]
        vec    = MPI_DIR_TO_VEC[mdir]
        target = (agent_pos[0] + vec[0], agent_pos[1] + vec[1])

        if target in walls or target == hp:
            # Wall or heavy box → single agent can't push heavy → no-op
            return [(1.0, agent_pos, sb0, sb1, hp)]

        if target in sboxes:
            # Small-box push attempt
            behind = (target[0] + vec[0], target[1] + vec[1])
            if blocked(behind, sboxes, hp):
                # Destination blocked → precondition not met → deterministic no-op
                return [(1.0, agent_pos, sb0, sb1, hp)]
            new_sb0 = behind if sb0 == target else sb0
            new_sb1 = behind if sb1 == target else sb1
            return [
                (push_p,       target,    new_sb0, new_sb1, hp),
                (1.0 - push_p, agent_pos, sb0,     sb1,     hp),
            ]

        # Move to free / goal cell — apply directional stochasticity
        dest_probs = {}
        for p, d in [(move_p, mdir),
                     (side_p, (mdir - 1) % 4),
                     (side_p, (mdir + 1) % 4)]:
            v    = MPI_DIR_TO_VEC[d]
            dest = (agent_pos[0] + v[0], agent_pos[1] + v[1])
            if blocked(dest, sboxes, hp):
                dest = agent_pos  # deviated into obstacle → stay
            dest_probs[dest] = dest_probs.get(dest, 0.0) + p

        return [(p, dest, sb0, sb1, hp) for dest, p in dest_probs.items()]

    # ── (NOOP, NOOP) ─────────────────────────────────────────────────────
    if a0_act == NOOP and a1_act == NOOP:
        return [(1.0, state, reward(box0_pos, box1_pos, heavy_pos))]

    # ── (dir, dir) — coordinated heavy-box push ───────────────────────────
    if a0_act != NOOP and a1_act != NOOP:
        if heavy_pos is None:
            return [(1.0, state, reward(box0_pos, box1_pos, heavy_pos))]

        mdir = ACTION_TO_DIR[a0_act]
        vec  = MPI_DIR_TO_VEC[mdir]
        # Both agents must be at the cell directly behind the heavy box
        required = (heavy_pos[0] - vec[0], heavy_pos[1] - vec[1])
        if a0_pos != required or a1_pos != required:
            return [(1.0, state, reward(box0_pos, box1_pos, heavy_pos))]

        new_hp  = (heavy_pos[0] + vec[0], heavy_pos[1] + vec[1])
        sboxes  = frozenset(p for p in (box0_pos, box1_pos) if p is not None)
        if blocked(new_hp, sboxes, None):
            return [(1.0, state, reward(box0_pos, box1_pos, heavy_pos))]

        # On success both agents advance to the old heavy-box cell
        success = (heavy_pos, heavy_pos, box0_pos, box1_pos, new_hp)
        return [
            (push_p,       success, reward(box0_pos, box1_pos, new_hp)),
            (1.0 - push_p, state,   reward(box0_pos, box1_pos, heavy_pos)),
        ]

    # ── Exactly one agent acts ────────────────────────────────────────────
    if a0_act != NOOP:
        raw = single_outcomes(a0_pos, a0_act, box0_pos, box1_pos, heavy_pos)
        return [
            (p, (np_, a1_pos, s0, s1, hp), reward(s0, s1, hp))
            for p, np_, s0, s1, hp in raw
        ]
    else:
        raw = single_outcomes(a1_pos, a1_act, box0_pos, box1_pos, heavy_pos)
        return [
            (p, (a0_pos, np_, s0, s1, hp), reward(s0, s1, hp))
            for p, np_, s0, s1, hp in raw
        ]


def build_transition_model(env):
    """
    Build the full MDP transition model by BFS over all reachable states.

    Returns
    -------
    transitions : dict
        transitions[state][joint_action] = [(prob, next_state, reward), ...]
    """
    transitions = {}

    env.reset()
    start = get_state(env)

    queue   = deque([start])
    visited = {start}

    logger.info("Building transition model via BFS...")
    expanded = 0
    while queue:
        state = queue.popleft()
        transitions[state] = {}
        expanded += 1

        for joint_action in JOINT_ACTIONS:
            outcomes = get_stochastic_outcomes(env, state, tuple(joint_action))
            transitions[state][tuple(joint_action)] = outcomes

            for _prob, next_state, _reward in outcomes:
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)

        if expanded % 500 == 0:
            logger.info("  expanded=%d  queue=%d  visited=%d",
                        expanded, len(queue), len(visited))

    logger.info("Done. Total states: %d", len(transitions))
    return transitions


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
    logger.info("Running MPI on %d states  (k=%d, gamma=%s)", len(states), k, gamma)

    V      = {s: 0.0 for s in states}
    policy = {s: JOINT_ACTIONS[0] for s in states}

    def q_value(s, ja):
        return sum(
            p * (r + gamma * V[ns])
            for p, ns, r in transitions[s][tuple(ja)]
        )

    for outer_iter in range(max_outer_iters):
        # Partial policy evaluation (k in-place sweeps)
        for _ in range(k):
            for s in states:
                V[s] = q_value(s, policy[s])

        # Policy improvement
        policy_stable = True
        changes = 0
        for s in states:
            best = max(JOINT_ACTIONS, key=lambda ja: q_value(s, ja))
            if tuple(best) != tuple(policy[s]):
                policy[s]    = best
                policy_stable = False
                changes += 1

        logger.info("  MPI iter %d: %d policy changes", outer_iter + 1, changes)

        if policy_stable:
            logger.info("Converged after %d iterations.", outer_iter + 1)
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

    logger.info("Starting solution_ex2 evaluation")
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    # ── Part 1: Online Planning ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part 1 — Online Planning (classical planner on stochastic env)")
    logger.info("=" * 60)

    # Direct evaluation loop for online planning
    online_steps = []
    t0 = time.time()
    for i in range(100):
        ep_t0 = time.time()
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        logger.info("  run %d/100 — steps: %d  (%.1fs)", i + 1, steps, time.time() - ep_t0)

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    logger.info("Online Planning  →  mean = %.2f  std = %.2f  (total %.1fs)",
                mean_ol, std_ol, time.time() - t0)

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part 2 — Modified Policy Iteration")
    logger.info("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    t1 = time.time()
    policy, V = modified_policy_iteration(env_mpi)
    logger.info("MPI model + solve took %.1fs", time.time() - t1)

    def mpi_policy_fn(env, obs):
        """Convert abstract MPI joint action to one real env step per agent."""
        state = get_state(env)
        if state not in policy:
            return {}
        joint_action = policy[state]
        agents = env.possible_agents
        step_actions = {}
        for i, agent in enumerate(agents):
            abstract_act = joint_action[i]
            if abstract_act == NOOP:
                continue
            target_minidir = ACTION_TO_DIR[abstract_act]
            current_dir    = env.agent_dirs[agent]
            if current_dir == target_minidir:
                step_actions[agent] = 2  # forward / push
            else:
                diff = (target_minidir - current_dir) % 4
                step_actions[agent] = 0 if diff == 3 else 1  # rotate left or right
        return step_actions

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    logger.info("MPI              →  mean = %.2f  std = %.2f", mean_mpi, std_mpi)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("%-25s %12s %12s", "Algorithm", "Mean steps", "Std steps")
    logger.info("-" * 50)
    logger.info("%-25s %12.2f %12.2f", "Online Planning", mean_ol, std_ol)
    logger.info("%-25s %12.2f %12.2f", "MPI", mean_mpi, std_mpi)
