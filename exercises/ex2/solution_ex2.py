"""
Assignment 2 — Probabilistic Box Pushing
"""
from __future__ import annotations

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
    get_environment().credits_stream = None
except ImportError:
    pass

_log_path = os.path.join(os.path.dirname(__file__), "solution_ex2.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("solution_ex2")

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

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

def _generate_plan(env: StochasticMultiAgentBoxPushEnv):
    domain_path, problem_path = generate_pddl_for_env(env)
    return solve_pddl(domain_path, problem_path)


def _extract_first_action_targets(plan) -> dict[str, tuple[int, int]] | None:
    if not plan or len(plan.actions) == 0:
        return None
    return extract_target_pos(plan.actions[0])


def _build_action_queues(
    env: StochasticMultiAgentBoxPushEnv,
    agent_targets: dict[str, tuple[int, int]],
) -> tuple[list[str], dict[str, list[int]]]:
    agents = list(agent_targets.keys())
    queues = {a: get_required_actions(env, a, agent_targets[a]) for a in agents}
    return agents, queues


def _pad_queues(agents: list[str], queues: dict[str, list[int | None]]) -> None:
    max_len = max(len(q) for q in queues.values())
    for a in agents:
        queues[a] = [None] * (max_len - len(queues[a])) + queues[a]


def _execute_queued_actions(
    env: StochasticMultiAgentBoxPushEnv,
    agents: list[str],
    queues: dict[str, list[int | None]],
) -> tuple[int, bool]:
    steps = 0
    done = False
    while any(len(queues[a]) > 0 for a in agents):
        step_actions: dict[str, int] = {}
        for a in agents:
            if queues[a]:
                act = queues[a].pop(0)
                if act is not None:
                    step_actions[a] = act
        _, _, terms, truncs, _ = env.step(step_actions)
        steps += 1
        if any(terms.values()) or any(truncs.values()):
            done = True
            break
    return steps, done


def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Replan from current state, execute only the first PDDL action, repeat.
    Returns total env steps taken.
    """
    env.reset()
    total_steps = 0
    done = False

    for _ in range(max_replans):
        if done:
            break

        plan = _generate_plan(env)
        targets = _extract_first_action_targets(plan)
        if not targets:
            break

        try:
            agents, queues = _build_action_queues(env, targets)
        except ValueError:
            continue

        _pad_queues(agents, queues)
        steps, done = _execute_queued_actions(env, agents, queues)
        total_steps += steps

    return total_steps


# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

Pos = tuple[int, int]
State = tuple[Pos, Pos, Pos | None, Pos | None, Pos | None]
JointAction = tuple[int, int]
Outcome = tuple[float, State, float]
SingleOutcome = tuple[float, Pos, Pos | None, Pos | None, Pos | None]

STAY  = 0
UP    = 1
RIGHT = 2
DOWN  = 3
LEFT  = 4

_ABSTRACT_TO_MINIDIR = {UP: 3, RIGHT: 0, DOWN: 1, LEFT: 2}
_MINIDIR_DELTA       = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_DIRECTIONS          = [UP, RIGHT, DOWN, LEFT]

JOINT_ACTIONS = (
    [(d, STAY) for d in _DIRECTIONS] +
    [(STAY, d) for d in _DIRECTIONS] +
    [(STAY, STAY)] +
    [(d, d) for d in _DIRECTIONS]
)


def _extract_walls(ascii_map: list[str]) -> set[Pos]:
    return {
        (x, y) for y, row in enumerate(ascii_map)
        for x, ch in enumerate(row) if ch == 'W'
    }


def _scan_boxes(env: StochasticMultiAgentBoxPushEnv) -> tuple[list[Pos], list[Pos]]:
    smalls: list[Pos] = []
    heavies: list[Pos] = []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                (heavies if getattr(cell, "box_size", "") == "heavy"
                 else smalls).append((x, y))
    smalls.sort()
    heavies.sort()
    return smalls, heavies


def _make_small_set(sb0: Pos | None, sb1: Pos | None) -> set[Pos]:
    return {p for p in (sb0, sb1) if p is not None}


def _add_pos(pos: Pos, delta: tuple[int, int]) -> Pos:
    return (pos[0] + delta[0], pos[1] + delta[1])


def _front_cell(pos: Pos, abstract_act: int) -> tuple[Pos, int]:
    minidir = _ABSTRACT_TO_MINIDIR[abstract_act]
    return _add_pos(pos, _MINIDIR_DELTA[minidir]), minidir


def _is_blocked(pos: Pos, walls: set[Pos], small_set: set[Pos], heavy: Pos | None) -> bool:
    return pos in walls or pos in small_set or pos == heavy


def _push_destination(box_pos: Pos, minidir: int) -> Pos:
    return _add_pos(box_pos, _MINIDIR_DELTA[minidir])


def _update_small_boxes(sb0: Pos | None, sb1: Pos | None, old_pos: Pos, new_pos: Pos) -> tuple[Pos | None, Pos | None]:
    return (new_pos if sb0 == old_pos else sb0,
            new_pos if sb1 == old_pos else sb1)


def _compute_goal_reward(sb0: Pos | None, sb1: Pos | None, heavy: Pos | None, goals: set[Pos]) -> float:
    placed = {p for p in (sb0, sb1, heavy) if p is not None}
    return 1.0 if goals <= placed else 0.0


def _slip_directions(minidir: int) -> list[int]:
    return [(minidir - 1) % 4, (minidir + 1) % 4]


def _resolve_free_move(
    agent_xy: Pos, minidir: int, move_p: float,
    walls: set[Pos], small_set: set[Pos], heavy: Pos | None,
) -> dict[Pos, float]:
    slip_p = (1.0 - move_p) / 2.0
    results: dict[Pos, float] = {}
    for prob, d in [(move_p, minidir)] + [(slip_p, sd) for sd in _slip_directions(minidir)]:
        dest = _add_pos(agent_xy, _MINIDIR_DELTA[d])
        if _is_blocked(dest, walls, small_set, heavy):
            dest = agent_xy
        results[dest] = results.get(dest, 0.0) + prob
    return results


def _resolve_small_push(
    agent_xy: Pos, front: Pos, minidir: int,
    sb0: Pos | None, sb1: Pos | None, heavy: Pos | None,
    walls: set[Pos], small_set: set[Pos], push_p: float,
) -> list[SingleOutcome]:
    behind = _push_destination(front, minidir)
    if _is_blocked(behind, walls, small_set, heavy):
        return [(1.0, agent_xy, sb0, sb1, heavy)]
    nsb0, nsb1 = _update_small_boxes(sb0, sb1, front, behind)
    return [
        (push_p,       front,    nsb0, nsb1, heavy),
        (1.0 - push_p, agent_xy, sb0,  sb1,  heavy),
    ]


def _resolve_single_agent(
    agent_xy: Pos, abstract_act: int,
    sb0: Pos | None, sb1: Pos | None, heavy: Pos | None,
    walls: set[Pos], push_p: float, move_p: float,
) -> list[SingleOutcome]:
    if abstract_act == STAY:
        return [(1.0, agent_xy, sb0, sb1, heavy)]

    front, minidir = _front_cell(agent_xy, abstract_act)
    small_set = _make_small_set(sb0, sb1)

    if front in walls or front == heavy:
        return [(1.0, agent_xy, sb0, sb1, heavy)]

    if front in small_set:
        return _resolve_small_push(agent_xy, front, minidir, sb0, sb1, heavy,
                                    walls, small_set, push_p)

    dests = _resolve_free_move(agent_xy, minidir, move_p, walls, small_set, heavy)
    return [(p, d, sb0, sb1, heavy) for d, p in dests.items()]


def _resolve_heavy_push(
    a0: Pos, a1: Pos, heavy: Pos,
    sb0: Pos | None, sb1: Pos | None,
    act0: int, walls: set[Pos], push_p: float,
) -> Pos | None:
    minidir = _ABSTRACT_TO_MINIDIR[act0]
    delta = _MINIDIR_DELTA[minidir]
    required = (heavy[0] - delta[0], heavy[1] - delta[1])
    if a0 != required or a1 != required:
        return None
    new_h = _add_pos(heavy, delta)
    sset = _make_small_set(sb0, sb1)
    if _is_blocked(new_h, walls, sset, None):
        return None
    return new_h


def _build_state_from_single(raw: list[SingleOutcome], other_pos: Pos, agent_idx: int,) -> list[tuple[float, State]]:
    results: list[tuple[float, State]] = []
    for p, na, s0, s1, h in raw:
        if agent_idx == 0:
            results.append((p, (na, other_pos, s0, s1, h)))
        else:
            results.append((p, (other_pos, na, s0, s1, h)))
    return results


def get_state(env: StochasticMultiAgentBoxPushEnv) -> State:
    agents = env.possible_agents
    a0 = env.agent_positions[agents[0]]
    a1 = env.agent_positions[agents[1]]
    smalls, heavies = _scan_boxes(env)
    return (
        a0, a1,
        smalls[0] if len(smalls) > 0 else None,
        smalls[1] if len(smalls) > 1 else None,
        heavies[0] if heavies else None,
    )


def compute_outcomes(env: StochasticMultiAgentBoxPushEnv, state: State, joint_action: JointAction) -> list[Outcome]:
    a0, a1, sb0, sb1, heavy = state
    act0, act1 = joint_action

    walls  = _extract_walls(env.ascii_map)
    goals  = set(env.goal_positions)
    push_p = env.push_success_prob
    move_p = env.move_success_prob

    def _r(s0, s1, h):
        return _compute_goal_reward(s0, s1, h, goals)

    if act0 == STAY and act1 == STAY:
        return [(1.0, state, _r(sb0, sb1, heavy))]

    if act0 != STAY and act1 != STAY:
        if heavy is None:
            return [(1.0, state, _r(sb0, sb1, heavy))]
        new_h = _resolve_heavy_push(a0, a1, heavy, sb0, sb1, act0, walls, push_p)
        if new_h is None:
            return [(1.0, state, _r(sb0, sb1, heavy))]
        ok = (heavy, heavy, sb0, sb1, new_h)
        return [
            (push_p,       ok,    _r(sb0, sb1, new_h)),
            (1.0 - push_p, state, _r(sb0, sb1, heavy)),
        ]

    if act0 != STAY:
        raw = _resolve_single_agent(a0, act0, sb0, sb1, heavy, walls, push_p, move_p)
        pairs = _build_state_from_single(raw, a1, 0)
        return [(p, ns, _r(ns[2], ns[3], ns[4])) for p, ns in pairs]
    else:
        raw = _resolve_single_agent(a1, act1, sb0, sb1, heavy, walls, push_p, move_p)
        pairs = _build_state_from_single(raw, a0, 1)
        return [(p, ns, _r(ns[2], ns[3], ns[4])) for p, ns in pairs]


def _bfs_expand(env: StochasticMultiAgentBoxPushEnv, start: State) -> dict[State, dict[JointAction, list[Outcome]]]:
    transitions = {}
    frontier = deque([start])
    seen = {start}
    count = 0
    while frontier:
        s = frontier.popleft()
        transitions[s] = {}
        count += 1
        for ja in JOINT_ACTIONS:
            key = tuple(ja)
            outcomes = compute_outcomes(env, s, key)
            transitions[s][key] = outcomes
            for _, ns, _ in outcomes:
                if ns not in seen:
                    seen.add(ns)
                    frontier.append(ns)
        if count % 500 == 0:
            logger.info("  expanded: %d  queued: %d", count, len(frontier))
    return transitions


def build_transition_model(env: StochasticMultiAgentBoxPushEnv) -> dict[State, dict[JointAction, list[Outcome]]]:
    env.reset()
    start = get_state(env)
    logger.info("Building transition model (BFS)...")
    transitions = _bfs_expand(env, start)
    logger.info("Done — %d states", len(transitions))
    return transitions


def _compute_q(T: dict, V: dict[State, float], gamma: float, s: State, ja: JointAction) -> float:
    return sum(p * (r + gamma * V[ns]) for p, ns, r in T[s][tuple(ja)])


def _partial_eval(T: dict, V: dict[State, float], policy: dict[State, JointAction], states: list[State], gamma: float, k: int) -> None:
    for _ in range(k):
        for s in states:
            V[s] = _compute_q(T, V, gamma, s, policy[s])


def _improve_policy(T: dict, V: dict[State, float], policy: dict[State, JointAction], states: list[State], gamma: float) -> int:
    n_changed = 0
    for s in states:
        best = max(JOINT_ACTIONS, key=lambda ja: _compute_q(T, V, gamma, s, ja))
        if tuple(best) != tuple(policy[s]):
            policy[s] = best
            n_changed += 1
    return n_changed


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration.
    Returns (policy, V) dicts mapping state -> joint_action and state -> float.
    """
    T: dict[State, dict[JointAction, list[Outcome]]] = build_transition_model(env)
    states: list[State] = list(T.keys())
    logger.info("MPI: %d states, k=%d, gamma=%.3f", len(states), k, gamma)

    V: dict[State, float] = {s: 0.0 for s in states}
    policy: dict[State, JointAction] = {s: JOINT_ACTIONS[0] for s in states}

    for iteration in range(1, max_outer_iters + 1):
        _partial_eval(T, V, policy, states, gamma, k)
        n_changed = _improve_policy(T, V, policy, states, gamma)
        logger.info("  iteration %d — %d changes", iteration, n_changed)
        if n_changed == 0:
            logger.info("Converged after %d iterations.", iteration)
            break

    return policy, V


# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================

def evaluate_policy(policy_fn, env: StochasticMultiAgentBoxPushEnv, n_runs: int = 100, max_steps: int = 500) -> tuple[float, float]:
    steps_per_run = []
    for _ in range(n_runs):
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done and steps < max_steps:
            actions = policy_fn(env, obs)
            obs, rewards, terms, truncs, _ = env.step(actions)
            steps += 1
            done = any(terms.values()) or any(truncs.values())
        steps_per_run.append(steps)
    return float(np.mean(steps_per_run)), float(np.std(steps_per_run))


# ===========================================================================
# Main
# ===========================================================================

def _translate_abstract_to_env_action(abstract_act: int, current_dir: int) -> int | None:
    if abstract_act == STAY:
        return None
    desired = _ABSTRACT_TO_MINIDIR[abstract_act]
    if current_dir == desired:
        return 2
    turn = (desired - current_dir) % 4
    return 0 if turn == 3 else 1


if __name__ == "__main__":
    logger.info("Starting solution_ex2 evaluation")
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    logger.info("=" * 60)
    logger.info("Part 1 — Online Planning")
    logger.info("=" * 60)

    online_steps = []
    t0 = time.time()
    for i in range(100):
        ep_t0 = time.time()
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        logger.info("  run %d/100 — steps: %d  (%.1fs)", i + 1, steps, time.time() - ep_t0)

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    logger.info("Online Planning → mean=%.2f std=%.2f (%.1fs total)",
                mean_ol, std_ol, time.time() - t0)

    logger.info("=" * 60)
    logger.info("Part 2 — Modified Policy Iteration")
    logger.info("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    t1 = time.time()
    policy, V = modified_policy_iteration(env_mpi)
    logger.info("MPI build+solve: %.1fs", time.time() - t1)

    def mpi_policy_fn(env, obs):
        state = get_state(env)
        if state not in policy:
            return {}
        ja = policy[state]
        agents = env.possible_agents
        actions = {}
        for idx, agent in enumerate(agents):
            act = _translate_abstract_to_env_action(ja[idx], env.agent_dirs[agent])
            if act is not None:
                actions[agent] = act
        return actions

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    logger.info("MPI → mean=%.2f std=%.2f", mean_mpi, std_mpi)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("%-25s %12s %12s", "Algorithm", "Mean steps", "Std steps")
    logger.info("-" * 50)
    logger.info("%-25s %12.2f %12.2f", "Online Planning", mean_ol, std_ol)
    logger.info("%-25s %12.2f %12.2f", "MPI", mean_mpi, std_mpi)
