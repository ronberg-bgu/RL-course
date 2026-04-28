"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

import random
import sys
import os
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

import unified_planning as up
up.shortcuts.get_environment().credits_stream = None

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

def _is_executable(pddl_action, env) -> bool:
    """
    Returns True if `pddl_action` can actually be executed in the env, False
    if it's a planner-emitted action that's physically impossible.

    Two failure modes covered:
      1. L-shape push: the planner allows pushes where (from, box, to) is not
         a straight line, but the env always pushes in the agent's facing
         direction (a straight shove).
      2. Move into an occupied cell: PDDL's `move` only checks adjacency, not
         whether `to` is clear, so the planner can emit moves into box cells.
         The env interprets such a "move" as a push attempt (and usually fails
         silently) — the agent can't actually land there.
    """
    action_str = str(pddl_action)
    locs = re.findall(r"loc_(\d+)_(\d+)", action_str)

    def _has_box(pos):
        x, y = pos
        cell = env.core_env.grid.get(x, y)
        return cell is not None and cell.type == "box"

    if action_str.startswith("move") and len(locs) >= 2:
        # move(agent, from, to)
        to = (int(locs[1][0]), int(locs[1][1]))
        return not _has_box(to)

    if "push" in action_str and len(locs) >= 3:
        fr  = (int(locs[0][0]), int(locs[0][1]))
        via = (int(locs[1][0]), int(locs[1][1]))
        to  = (int(locs[2][0]), int(locs[2][1]))
        # Straight-line check
        return (via[0] - fr[0], via[1] - fr[1]) == (to[0] - via[0], to[1] - via[1])

    return True  # unknown action, give it the benefit of the doubt

def run_online_planning(env, max_replans: int = 500) -> int:
    """
    Online planning: at each iteration, replan from the current state and
    execute the first PDDL macro-action.
 
    Two safety mechanisms protect against the PDDL/env mismatch:
      1. L-shape pushes are detected and skipped — the planner can produce
         them (only adjacency is checked) but the env can never execute them.
      2. If the same first-action keeps coming back, we assume the env state
         is not changing and inject a random perturbation to escape.
    """
    obs, _ = env.reset()
    total_env_steps = 0
    done = False
    last_action_str = None
    stuck_count = 0
 
    for _ in range(max_replans):
        if done:
            break
 
        domain_path, problem_path = generate_pddl_for_env(env)
        plan = solve_pddl(domain_path, problem_path)
        if not plan or len(plan.actions) == 0:
            break
 
        pddl_action = plan.actions[0]
        action_str = str(pddl_action)
 
        # ── Progress detection ──────────────────────────────────────────
        # No-progress signals:
        #   (a) Planner emitted an L-shape push (impossible in env)
        #   (b) First action of the new plan matches the previous one
        l_shape = not _is_executable(pddl_action, env)
        same_as_before = (action_str == last_action_str)
 
        if l_shape or same_as_before:
            stuck_count += 1
        else:
            stuck_count = 0
        last_action_str = action_str
 
        # L-shape pushes will never work, so escalate immediately
        if l_shape:
            stuck_count = max(stuck_count, 3)
 
        # ── Perturbation: random action to break out of a deadlock ──────
        if stuck_count >= 3:
            step_actions = {a: random.choice([0, 1, 2]) for a in env.possible_agents}
            _, _, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            stuck_count = 0
            last_action_str = None
            continue
 
        # ── Translate first PDDL action into env-level action queue ─────
        agent_targets = extract_target_pos(pddl_action)
        if not agent_targets:
            break
        agents_in_action = list(agent_targets.keys())
 
        try:
            action_queues = {
                a: get_required_actions(env, a, agent_targets[a])
                for a in agents_in_action
            }
        except ValueError:
            # Target became non-adjacent due to a slip → force a perturbation
            stuck_count += 3
            continue
 
        max_len = max(
            (len(q) for q in action_queues.values() if q is not None), default=0
        )
        for a in agents_in_action:
            if action_queues[a] is None:
                action_queues[a] = []
            action_queues[a] = (
                [None] * (max_len - len(action_queues[a])) + action_queues[a]
            )
 
        # ── Execute the macro-action one env-step at a time ─────────────
        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {agent: 4 for agent in env.possible_agents}
            for a in agents_in_action:
                if action_queues[a]:
                    act = action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act
 
            _, _, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
                break
 
    return total_env_steps

# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

from collections import defaultdict, deque

# Cardinal actions: 0=N, 1=S, 2=E, 3=W, 4=Stay
N_, S_, E_, W_, STAY = 0, 1, 2, 3, 4
DELTAS = {N_: (0, -1), S_: (0, 1), E_: (1, 0), W_: (-1, 0), STAY: (0, 0)}

# Deviations in screen coords (y grows downward):
#   N → left=W, right=E   |   S → left=E, right=W
#   E → left=N, right=S   |   W → left=S, right=N
DEVIATIONS = {N_: (W_, E_), S_: (E_, W_), E_: (N_, S_), W_: (S_, N_)}

MOVE_OK, MOVE_DEV = 0.8, 0.1
PUSH_OK, PUSH_FAIL = 0.8, 0.2

STEP_REWARD = -1.0
GOAL_REWARD = 0.0

# Goal hardcoded from problem.pddl for this map:
#   small boxes cover {(2,5), (4,5)}, heavy box at (6,5)
GOAL_SMALL = frozenset({(2, 5), (4, 5)})
GOAL_HEAVY = (6, 5)


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
def get_state_info(env):
    """Returns the sorted state, and a boolean indicating if agents were swapped."""
    agents = env.possible_agents
    p0 = tuple(env.agent_positions[agents[0]])
    p1 = tuple(env.agent_positions[agents[1]])

    is_swapped = p0 > p1
    a_pos = (p1, p0) if is_swapped else (p0, p1)

    small, heavy = [], None
    for y in range(env.height):
        for x in range(env.width):
            c = env.core_env.grid.get(x, y)
            if c is not None and c.type == "box":
                if getattr(c, "box_size", "") == "heavy":
                    heavy = (x, y)
                else:
                    small.append((x, y))

    state = (a_pos[0], a_pos[1], frozenset(small), heavy)
    return state, is_swapped


def get_state(env):
    state, _ = get_state_info(env)
    return state


def _initial_state(env):
    state, _ = get_state_info(env)
    return state


def _walkable(env):
    cells = set()
    for y in range(env.height):
        for x in range(env.width):
            c = env.core_env.grid.get(x, y)
            if c is not None and c.type == "wall":
                continue
            cells.add((x, y))
    return cells


def _is_terminal(state):
    _, _, small_boxes, heavy_pos = state
    return small_boxes == GOAL_SMALL and heavy_pos == GOAL_HEAVY


def _occupant(pos, small_boxes, heavy_pos):
    if pos == heavy_pos:
        return "heavy"
    if pos in small_boxes:
        return "small"
    return None


def _move_with_deviations(agent_pos, direction, small_boxes, heavy_pos,
                          walkable, other_pos):
    """0.8/0.1/0.1 move; boxes block, but other agent does NOT block (overlap allowed)."""
    outcomes = []
    left_dir, right_dir = DEVIATIONS[direction]
    for p, d in [(MOVE_OK, direction), (MOVE_DEV, left_dir), (MOVE_DEV, right_dir)]:
        ddx, ddy = DELTAS[d]
        tgt = (agent_pos[0] + ddx, agent_pos[1] + ddy)
        occ = _occupant(tgt, small_boxes, heavy_pos)
        blocked = (tgt not in walkable) or (occ is not None)
        new_pos = agent_pos if blocked else tgt
        outcomes.append((p, new_pos, small_boxes, heavy_pos))
    return outcomes


def _agent_outcomes(agent_pos, action, other_pos, small_boxes, heavy_pos, walkable):
    if action == STAY:
        return [(1.0, agent_pos, small_boxes, heavy_pos)]

    dx, dy = DELTAS[action]
    target = (agent_pos[0] + dx, agent_pos[1] + dy)
    occ = _occupant(target, small_boxes, heavy_pos)

    # Push-small attempt
    if occ == "small":
        beyond = (target[0] + dx, target[1] + dy)
        # Agent in 'beyond' cell still blocks the box from moving there
        beyond_blocked = (beyond not in walkable or beyond in small_boxes or 
                          beyond == heavy_pos or beyond == other_pos)
        if beyond_blocked:
            return [(1.0, agent_pos, small_boxes, heavy_pos)]
        
        new_small = set(small_boxes)
        new_small.remove(target)
        new_small.add(beyond)
        return [(PUSH_OK, target, frozenset(new_small), heavy_pos),
                (PUSH_FAIL, agent_pos, small_boxes, heavy_pos)]

    # Heavy box blocks single agents
    if occ == "heavy":
        return [(1.0, agent_pos, small_boxes, heavy_pos)]

    # Normal move (using your 0.8/0.1/0.1 logic)
    return _move_with_deviations(agent_pos, action, small_boxes, heavy_pos, walkable, other_pos)


def _joint_transition(state, joint_action, walkable):
    """Return list of (prob, next_state) for this joint action."""
    a0_pos, a1_pos, small_boxes, heavy_pos = state
    a0_act, a1_act = joint_action

    # Heavy push: both agents co-located, same non-STAY direction, target = heavy
    if a0_pos == a1_pos and a0_act == a1_act and a0_act != STAY:
        dx, dy = DELTAS[a0_act]
        target = (a0_pos[0] + dx, a0_pos[1] + dy)
        if _occupant(target, small_boxes, heavy_pos) == "heavy":
            beyond = (target[0] + dx, target[1] + dy)
            beyond_blocked = (
                beyond not in walkable
                or beyond in small_boxes
            )
            if beyond_blocked:
                return [(1.0, state)]
            return [
                (PUSH_OK,   (target, target, small_boxes, beyond)),
                (PUSH_FAIL, state),
            ]

    # General independent-then-combine case
    a0_outs = _agent_outcomes(a0_pos, a0_act, a1_pos, small_boxes, heavy_pos, walkable)
    a1_outs = _agent_outcomes(a1_pos, a1_act, a0_pos, small_boxes, heavy_pos, walkable)

    final_combined = defaultdict(float)
    for p0, na0, sb0, hp0 in a0_outs:
        for p1, na1, sb1, hp1 in a1_outs:
            prob = p0 * p1

            # Merge small-box changes
            if sb0 != small_boxes and sb1 != small_boxes:
                moved_by_0 = (small_boxes - sb0) | (sb0 - small_boxes)
                moved_by_1 = (small_boxes - sb1) | (sb1 - small_boxes)
                if moved_by_0 & moved_by_1:
                    new_small = small_boxes
                    na0, na1 = a0_pos, a1_pos
                else:
                    new_small = frozenset(
                        (small_boxes - moved_by_0 - moved_by_1)
                        | (sb0 - small_boxes)
                        | (sb1 - small_boxes)
                    )
            elif sb0 != small_boxes: new_small = sb0
            elif sb1 != small_boxes: new_small = sb1
            else: new_small = small_boxes

            new_heavy = hp0 if hp0 != heavy_pos else (hp1 if hp1 != heavy_pos else heavy_pos)

            # SORT AGENTS IMMEDIATELY BEFORE HASHING TO PREVENT DUPLICATES
            if na0 > na1:
                na0, na1 = na1, na0
                
            final_combined[(na0, na1, new_small, new_heavy)] += prob

    return [(prob, s) for s, prob in final_combined.items()]

def is_dead_end(state):
    _, _, small_boxes, heavy_pos = state
    
    # 1. Top Wall (y=1) and Left Wall (x=1) are universally dead.
    if heavy_pos[1] == 1 or heavy_pos[0] == 1: return True
    for sb in small_boxes:
        if sb[1] == 1 or sb[0] == 1: return True
        
    # 2. Small boxes should NEVER touch the right wall (x=6) 
    for sb in small_boxes:
        if sb[0] == 6: return True

    # 3. Heavy box should NEVER go left of x=3. 
    if heavy_pos[0] < 3: return True
    
    # 4. ANTI-REGRESSION HEURISTIC
    # The heavy box starts at (4,2) and needs to reach (6,5).
    # Its starting Manhattan distance to the goal is |4-6| + |2-5| = 5.
    # If the agents push it so badly that its distance becomes > 6, 
    # they are going the wrong way. Snip the branch!
    dist_to_heavy_goal = abs(heavy_pos[0] - 6) + abs(heavy_pos[1] - 5)
    if dist_to_heavy_goal > 6:
        return True
        
    return False

def build_transition_model(env):
    """
    BFS over reachable states, computing transitions[s][joint_action] =
    [(prob, next_state, reward), ...].
    """
    walkable = _walkable(env)
    initial = _initial_state(env)
    joint_actions = [(i, j) for i in range(5) for j in range(5)]

    transitions = {}
    visited = {initial}
    queue = deque([initial])
    terminals = set()

    while queue:
        s = queue.popleft()

        if _is_terminal(s):
            terminals.add(s)
            transitions[s] = {a: [(1.0, s, GOAL_REWARD)] for a in joint_actions}
            continue

        transitions[s] = {}
        for a in joint_actions:
            outs = _joint_transition(s, a, walkable)
            valid_triples = []
            
            for p, s_next in outs:
                # If an action pushes a box into a dead end, treat the action as a failure 
                # (the agent stays in state 's' and eats the -1 reward penalty).
                if is_dead_end(s_next):
                    valid_triples.append((p, s, STEP_REWARD))
                else:
                    valid_triples.append((p, s_next, STEP_REWARD))
                    if s_next not in visited:
                        visited.add(s_next)
                        queue.append(s_next)
                        
            transitions[s][a] = valid_triples

    return transitions, list(visited), terminals

def _heuristic_value(state):
    """
    Estimates the negative reward (distance) to the goals.
    Since every step costs -1, the value of a state is roughly 
    the negative distance of the boxes to their goals.
    """
    _, _, small_boxes, heavy_pos = state
    
    # Distance for the heavy box to (6, 5)
    h_dist = abs(heavy_pos[0] - 6) + abs(heavy_pos[1] - 5)
    
    # Distance for small boxes to {(2, 5), (4, 5)}
    s_dist = 0
    goals = [(2, 5), (4, 5)]
    for sb in small_boxes:
        # Find the distance to the closest small goal
        s_dist += min(abs(sb[0] - gx) + abs(sb[1] - gy) for gx, gy in goals)
        
    # Return negative total distance (because steps are -1 reward)
    return float(-(h_dist + s_dist))

def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration.
    """
    # 1. Build the model once
    print("Building transition model for MPI...")
    transitions, states, terminals = build_transition_model(env)
    
    # 2. Initialize V and Policy
    V = {s: _heuristic_value(s) for s in states}
    joint_actions = [(i, j) for i in range(5) for j in range(5)]
    # Default policy: both agents STAY (action 4)
    policy = {s: (4, 4) for s in states}

    for i in range(max_outer_iters):
        # --- STEP A: Partial Policy Evaluation (k sweeps) ---
        for _ in range(k):
            delta = 0
            for s in states:
                if s in terminals: continue
                
                old_v = V[s]
                action = policy[s]
                
                v_new = 0.0
                for p, sn, r in transitions[s][action]:
                    v_new += p * (r + gamma * V[sn])
                
                V[s] = v_new
                diff = abs(old_v - v_new)
                if diff > delta: 
                    delta = diff
            
            if delta < theta:
                break

        # --- STEP B: Policy Improvement ---
        policy_stable = True
        for s in states:
            if s in terminals: continue
            
            old_action = policy[s]
            best_q = -float('inf')
            best_action = old_action

            for a in joint_actions:
                q_sa = 0.0
                for p, sn, r in transitions[s][a]:
                    q_sa += p * (r + gamma * V[sn])
                    
                if q_sa > best_q:
                    best_q = q_sa
                    best_action = a
            
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        
        if policy_stable:
            print(f"MPI converged at iteration {i}")
            break
            
    return policy, V


# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================

def evaluate_policy(policy_fn, env, n_runs: int = 100, max_steps: int = 500):
    """
    Run *policy_fn* for n_runs episodes and return (mean_steps, std_steps).
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
    print("Part 1 - Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    # Wrap run_online_planning as a policy function for the evaluator
    def online_planning_policy(env, obs):
        raise NotImplementedError(
            "Adapt this shim or call run_online_planning directly in a loop."
        )

    # Direct evaluation loop for online planning
    online_steps = []
    for i in range(2):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        print(f"  run {i+1}/2 - steps so far: {steps}, running mean: {np.mean(online_steps):.1f}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  ->  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 - Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)

    def mpi_policy_fn(env, obs):
        """Convert current env state to a joint action using the MPI policy."""
        state, is_swapped = get_state_info(env)
        act0, act1 = policy[state]
        
        agents = env.possible_agents
        # If agents were swapped to find the state, swap the actions back!
        if is_swapped:
            return {agents[0]: act1, agents[1]: act0}
        return {agents[0]: act0, agents[1]: act1}

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    print(f"\nMPI              ->  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")