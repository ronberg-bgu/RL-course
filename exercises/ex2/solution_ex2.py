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
import time
from collections import defaultdict, deque

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
    """
    action_str = str(pddl_action)
    locs = re.findall(r"loc_(\d+)_(\d+)", action_str)

    def _has_box(pos):
        x, y = pos
        cell = env.core_env.grid.get(x, y)
        return cell is not None and cell.type == "box"

    if action_str.startswith("move") and len(locs) >= 2:
        to = (int(locs[1][0]), int(locs[1][1]))
        return not _has_box(to)

    if "push" in action_str and len(locs) >= 3:
        fr  = (int(locs[0][0]), int(locs[0][1]))
        via = (int(locs[1][0]), int(locs[1][1]))
        to  = (int(locs[2][0]), int(locs[2][1]))
        return (via[0] - fr[0], via[1] - fr[1]) == (to[0] - via[0], to[1] - via[1])

    return True 

def run_online_planning(env, max_replans: int = 500) -> int:
    """
    Online planning: at each iteration, replan from the current state and
    execute the first PDDL macro-action.
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
 
        l_shape = not _is_executable(pddl_action, env)
        same_as_before = (action_str == last_action_str)
 
        if l_shape or same_as_before:
            stuck_count += 1
        else:
            stuck_count = 0
        last_action_str = action_str
 
        if l_shape:
            stuck_count = max(stuck_count, 3)
 
        if stuck_count >= 3:
            step_actions = {a: random.choice([0, 1, 2]) for a in env.possible_agents}
            _, _, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            stuck_count = 0
            last_action_str = None
            continue
 
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

# Cardinal actions: 0=N, 1=S, 2=E, 3=W, 4=Stay
N_, S_, E_, W_, STAY = 0, 1, 2, 3, 4
DELTAS = {N_: (0, -1), S_: (0, 1), E_: (1, 0), W_: (-1, 0), STAY: (0, 0)}

DEVIATIONS = {N_: (W_, E_), S_: (E_, W_), E_: (N_, S_), W_: (S_, N_)}

MOVE_OK, MOVE_DEV = 0.8, 0.1
PUSH_OK, PUSH_FAIL = 0.8, 0.2

STEP_REWARD = -1.0
GOAL_REWARD = 0.0

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

def _move_with_deviations(agent_pos, direction, small_boxes, heavy_pos, walkable, other_pos):
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

    if occ == "small":
        beyond = (target[0] + dx, target[1] + dy)
        beyond_blocked = (beyond not in walkable or beyond in small_boxes or 
                          beyond == heavy_pos or beyond == other_pos)
        if beyond_blocked:
            return [(1.0, agent_pos, small_boxes, heavy_pos)]
        
        new_small = set(small_boxes)
        new_small.remove(target)
        new_small.add(beyond)
        return [(PUSH_OK, target, frozenset(new_small), heavy_pos),
                (PUSH_FAIL, agent_pos, small_boxes, heavy_pos)]

    if occ == "heavy":
        return [(1.0, agent_pos, small_boxes, heavy_pos)]

    return _move_with_deviations(agent_pos, action, small_boxes, heavy_pos, walkable, other_pos)

def _joint_transition(state, joint_action, walkable):
    a0_pos, a1_pos, small_boxes, heavy_pos = state
    a0_act, a1_act = joint_action

    if a0_pos == a1_pos and a0_act == a1_act and a0_act != STAY:
        dx, dy = DELTAS[a0_act]
        target = (a0_pos[0] + dx, a0_pos[1] + dy)
        if _occupant(target, small_boxes, heavy_pos) == "heavy":
            beyond = (target[0] + dx, target[1] + dy)
            beyond_blocked = (beyond not in walkable or beyond in small_boxes)
            if beyond_blocked:
                return [(1.0, state)]
            return [
                (PUSH_OK,   (target, target, small_boxes, beyond)),
                (PUSH_FAIL, state),
            ]

    a0_outs = _agent_outcomes(a0_pos, a0_act, a1_pos, small_boxes, heavy_pos, walkable)
    a1_outs = _agent_outcomes(a1_pos, a1_act, a0_pos, small_boxes, heavy_pos, walkable)

    final_combined = defaultdict(float)
    for p0, na0, sb0, hp0 in a0_outs:
        for p1, na1, sb1, hp1 in a1_outs:
            prob = p0 * p1

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

            if na0 > na1:
                na0, na1 = na1, na0
                
            final_combined[(na0, na1, new_small, new_heavy)] += prob

    return [(prob, s) for s, prob in final_combined.items()]

def is_dead_end(state):
    _, _, small_boxes, heavy_pos = state
    if heavy_pos[1] == 1 or heavy_pos[0] == 1: return True
    for sb in small_boxes:
        if sb[1] == 1 or sb[0] == 1: return True
    for sb in small_boxes:
        if sb[0] == 6: return True
    if heavy_pos[0] < 3: return True
    dist_to_heavy_goal = abs(heavy_pos[0] - 6) + abs(heavy_pos[1] - 5)
    if dist_to_heavy_goal > 6: return True
    for sb in small_boxes:
        if sb[1] == 5 and sb not in GOAL_SMALL: return True
    if heavy_pos[1] == 5 and heavy_pos != GOAL_HEAVY: return True
    return False

def build_transition_model(env):
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
    _, _, small_boxes, heavy_pos = state
    h_dist = abs(heavy_pos[0] - 6) + abs(heavy_pos[1] - 5)
    s_dist = 0
    goals = [(2, 5), (4, 5)]
    for sb in small_boxes:
        s_dist += min(abs(sb[0] - gx) + abs(sb[1] - gy) for gx, gy in goals)
    return float(-(h_dist + s_dist))

def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    Modified Policy Iteration — vectorized with numpy for speed.
    """
    # 1. Build the model 
    t0 = time.time()
    print("Building transition model for MPI...", flush=True)
    transitions, states, terminals = build_transition_model(env)
    print(f"  built in {time.time()-t0:.1f}s — "
          f"{len(states):,} states, {len(terminals)} terminals", flush=True)

    # 2. Flatten transitions into numpy arrays 
    print("Vectorizing transition model...", flush=True)
    t1 = time.time()

    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    joint_actions = [(i, j) for i in range(5) for j in range(5)]
    n_actions = len(joint_actions)
    action_to_idx = {a: i for i, a in enumerate(joint_actions)}

    max_outcomes = max(
        len(transitions[s][a]) for s in states for a in joint_actions
    )

    P = np.zeros((n_states, n_actions, max_outcomes), dtype=np.float32)
    NS = np.zeros((n_states, n_actions, max_outcomes), dtype=np.int32)
    R = np.zeros((n_states, n_actions, max_outcomes), dtype=np.float32)

    for s in states:
        si = state_to_idx[s]
        for a_tup, ai in action_to_idx.items():
            triples = transitions[s][a_tup]
            for k_idx, (p, sn, r) in enumerate(triples):
                P[si, ai, k_idx] = p
                NS[si, ai, k_idx] = state_to_idx[sn]
                R[si, ai, k_idx] = r

    terminal_mask = np.zeros(n_states, dtype=bool)
    for t in terminals:
        terminal_mask[state_to_idx[t]] = True
    nonterminal_idx = np.where(~terminal_mask)[0]

    print(f"  vectorized in {time.time()-t1:.1f}s "
          f"(P shape {P.shape}, ~{P.nbytes/1e6:.0f}MB)", flush=True)

    # 3. Initialize V (with heuristic) and policy 
    V = np.array([_heuristic_value(s) for s in states], dtype=np.float32)
    V[terminal_mask] = 0.0  
    policy_idx = np.zeros(n_states, dtype=np.int32)  

    # 4. MPI loop, vectorized 
    print(f"Starting MPI (gamma={gamma}, k={k})...", flush=True)
    mpi_start = time.time()

    for outer in range(max_outer_iters):
        iter_start = time.time()

        # Step A: Partial policy evaluation (k sweeps)
        eval_sweeps_done = 0
        for sweep in range(k):
            P_pi = P[np.arange(n_states), policy_idx, :]
            NS_pi = NS[np.arange(n_states), policy_idx, :]
            R_pi = R[np.arange(n_states), policy_idx, :]

            V_new = np.sum(P_pi * (R_pi + gamma * V[NS_pi]), axis=1)
            V_new[terminal_mask] = 0.0  

            delta = np.max(np.abs(V_new - V))
            V = V_new
            eval_sweeps_done += 1
            if delta < theta:
                break

        # Step B: Policy improvement 
        Q = np.sum(P * (R + gamma * V[NS]), axis=2)
        new_policy_idx = np.argmax(Q, axis=1)
        new_policy_idx[terminal_mask] = 0  

        n_changed = int(np.sum(new_policy_idx[nonterminal_idx]
                               != policy_idx[nonterminal_idx]))
        policy_stable = (n_changed == 0)
        policy_idx = new_policy_idx

        iter_time = time.time() - iter_start
        print(f"  outer iter {outer+1:3d}: {eval_sweeps_done} eval sweeps, "
              f"{n_changed:,} policy changes, {iter_time:.1f}s "
              f"(total {time.time()-mpi_start:.1f}s)", flush=True)

        if policy_stable:
            print(f"MPI converged at iteration {outer+1}", flush=True)
            break

    # 5. Convert numpy policy back to dict 
    policy = {s: joint_actions[policy_idx[i]] for i, s in enumerate(states)}
    V_dict = {s: float(V[i]) for i, s in enumerate(states)}
    return policy, V_dict

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
    for i in range(100):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 10 == 0:
            print(f"  run {i+1}/100 — steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  ->  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 - Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    env_mpi.reset()
    policy, V = modified_policy_iteration(env_mpi)

    # MiniGrid direction encoding (DIR_TO_VEC ordering):
    #   0 = right (East), 1 = down (South), 2 = left (West), 3 = up (North)
    # Our MDP cardinals (DELTAS keys):
    #   0 = N (up),  1 = S (down),  2 = E (right),  3 = W (left),  4 = Stay
    CARDINAL_TO_MINIGRID = {
        0: 3,   # MPI N → MiniGrid up
        1: 1,   # MPI S → MiniGrid down
        2: 0,   # MPI E → MiniGrid right
        3: 2,   # MPI W → MiniGrid left
    }

    def mpi_policy_fn(env, obs):
        """
        Translate MPI's cardinal-direction policy into env-level actions.

        MPI policy outputs (cardinal_a0, cardinal_a1) where each is one of
        {0:N, 1:S, 2:E, 3:W, 4:Stay}. The env, however, takes per-agent
        actions {0: rotate-left, 1: rotate-right, 2: forward, 4: stay}.

        We translate:
          - if the agent already faces the target cardinal -> forward (2)
          - else -> rotate one step toward the target this tick (the policy
            stays the same on the next tick since rotation doesn't change
            the MDP state we modeled, so we'll keep rotating until aligned
            and then go forward)
        """
        state, is_swapped = get_state_info(env)
        a_tup = policy.get(state, (4, 4))
        if is_swapped:
            a_tup = (a_tup[1], a_tup[0])

        agents = env.possible_agents
        env_actions = {}

        for i, agent in enumerate(agents):
            cardinal = a_tup[i]
            if cardinal == 4:              # STAY
                env_actions[agent] = 4
                continue

            target_dir = CARDINAL_TO_MINIGRID[cardinal]
            cur_dir = env.agent_dirs[agent]

            if cur_dir == target_dir:
                env_actions[agent] = 2     # forward
            else:
                # Pick the shorter rotation
                cw_dist = (target_dir - cur_dir) % 4
                if cw_dist <= 2:
                    env_actions[agent] = 1  # rotate right (clockwise)
                else:
                    env_actions[agent] = 0  # rotate left

        return env_actions

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