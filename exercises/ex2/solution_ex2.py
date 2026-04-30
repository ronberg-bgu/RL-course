"""
Assignment 2 - Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  - online replanning loop
  2. build_transition_model - MDP transition model (used by MPI)
  3. modified_policy_iteration - MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

from gettext import translation
import itertools
import sys
import os
import contextlib
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions
from collections import deque
from minigrid.core.constants import DIR_TO_VEC
try:
    import unified_planning as up
    up.shortcuts.get_environment().credits_stream = None
except ImportError:
    pass

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
class BoxPushPhysicsEngine:
    """Handles all state transitions, probabilities, and physical rules of the environment."""
    
    def _init_(self, env):
        self.move_prob = getattr(env, "move_success_prob", 0.8)
        self.push_prob = getattr(env, "push_success_prob", 0.8)
        self.slip_prob = (1.0 - self.move_prob) / 2.0
        
        self.walkable = set()
        self.goals = set()
        self.vectors = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        
        for y in range(env.height):
            for x in range(env.width):
                cell = env.core_env.grid.get(x, y)
                if cell is None or cell.type != "wall":
                    self.walkable.add((x, y))
                if cell and cell.type == "goal":
                    self.goals.add((x, y))

    def extract_start_state(self, env) -> tuple:
        agents = tuple(sorted(tuple(env.agent_positions[a]) for a in env.possible_agents))
        smalls = []
        heavy = None
        
        for y in range(env.height):
            for x in range(env.width):
                c = env.core_env.grid.get(x, y)
                if c and c.type == "box":
                    if getattr(c, "box_size", "") == "heavy":
                        heavy = (x, y)
                    else:
                        smalls.append((x, y))
                        
        return (agents, frozenset(smalls), heavy)

    def is_victory(self, state) -> bool:
        if state == "DEAD_END": return False
        agents, smalls, heavy = state
        boxes = list(smalls) + ([heavy] if heavy else [])
        return len(boxes) > 0 and all(b in self.goals for b in boxes)

    def is_doomed(self, state) -> bool:
        if state == "DEAD_END": return True
        agents, smalls, heavy = state
        boxes = list(smalls) + ([heavy] if heavy else [])
        for box in boxes:
            if box[0] == 1 or box[1] == 1:
                return True
        return False

    def _resolve_single_agent(self, pos, action_dir, current_smalls, heavy_pos):
        """Resolves physical branching for a single agent."""
        tgt = (pos[0] + self.vectors[action_dir][0], pos[1] + self.vectors[action_dir][1])
        
        def evaluate(intent_tgt, intent_dir, is_push_intent):
            if intent_tgt not in self.walkable or intent_tgt == heavy_pos:
                return [(1.0 if is_push_intent else self.slip_prob, pos, current_smalls)]
                
            if intent_tgt in current_smalls:
                if not is_push_intent:
                    return [(self.slip_prob, pos, current_smalls)]
                
                push_tgt = (intent_tgt[0] + self.vectors[intent_dir][0], intent_tgt[1] + self.vectors[intent_dir][1])
                if push_tgt in self.walkable and push_tgt != heavy_pos and push_tgt not in current_smalls:
                    new_smalls = set(current_smalls)
                    new_smalls.remove(intent_tgt)
                    new_smalls.add(push_tgt)
                    return [
                        (self.push_prob, intent_tgt, frozenset(new_smalls)),
                        (1.0 - self.push_prob, pos, current_smalls)
                    ]
                else:
                    return [(1.0, pos, current_smalls)]
                    
            p = self.move_prob if is_push_intent else self.slip_prob
            return [(p, intent_tgt, current_smalls)]

        branches = []
        branches.extend(evaluate(tgt, action_dir, True))
        
        l_dir = (action_dir - 1) % 4
        l_tgt = (pos[0] + self.vectors[l_dir][0], pos[1] + self.vectors[l_dir][1])
        branches.extend(evaluate(l_tgt, l_dir, False))
        
        r_dir = (action_dir + 1) % 4
        r_tgt = (pos[0] + self.vectors[r_dir][0], pos[1] + self.vectors[r_dir][1])
        branches.extend(evaluate(r_tgt, r_dir, False))
        
        # Consolidate duplicate physical outcomes
        consolidated = {}
        for p, n_pos, n_smalls in branches:
            k = (n_pos, n_smalls)
            consolidated[k] = consolidated.get(k, 0.0) + p
            
        return [(p, k[0], k[1]) for k, p in consolidated.items()]

    def get_successors(self, state, joint_action):
        """Core physics engine utilizing Sequential Resolution."""
        if state == "DEAD_END":
            return [(1.0, "DEAD_END", 0.0)]
            
        agents, smalls, heavy = state
        
        a0, a1 = agents
        cmd0, cmd1 = joint_action
        
        tgt0 = (a0[0] + self.vectors[cmd0][0], a0[1] + self.vectors[cmd0][1])
        tgt1 = (a1[0] + self.vectors[cmd1][0], a1[1] + self.vectors[cmd1][1])
        
        is_joint_heavy = (tgt0 == tgt1 and tgt0 == heavy and a0 == a1 and cmd0 == cmd1)
        
        successors = {}
        
        if is_joint_heavy:
            push_dest = (tgt0[0] + self.vectors[cmd0][0], tgt0[1] + self.vectors[cmd0][1])
            if push_dest in self.walkable and push_dest not in smalls:
                succ_state = ((tgt0, tgt0), smalls, push_dest)
                fail_state = (agents, smalls, heavy)
                
                successors[succ_state if not self.is_doomed(succ_state) else "DEAD_END"] = self.push_prob
                successors[fail_state if not self.is_doomed(fail_state) else "DEAD_END"] = 1.0 - self.push_prob
            else:
                successors[state] = 1.0
        else:
            # Sequential Branching Strategy
            intermediate_states = self._resolve_single_agent(a0, cmd0, smalls, heavy)
            
            for p0, inter_a0, inter_smalls in intermediate_states:
                final_branches = self._resolve_single_agent(a1, cmd1, inter_smalls, heavy)
                
                for p1, final_a1, final_smalls in final_branches:
                    n_state = (tuple(sorted([inter_a0, final_a1])), final_smalls, heavy)
                    
                    if self.is_doomed(n_state):
                        n_state = "DEAD_END"
                        
                    successors[n_state] = successors.get(n_state, 0.0) + (p0 * p1)
                    
        return [(prob, s, 1.0 if self.is_victory(s) else 0.0) for s, prob in successors.items()]

# ===========================================================================
# Part 1 - Online Planning
# ===========================================================================

import unified_planning as up

def run_online_planning(env, max_replans: int = 300) -> int:
    obs, _ = env.reset()
    steps_taken = 0
    is_terminal = False

    for _ in range(max_replans):
        if is_terminal:
            break

        domain_path, problem_path = generate_pddl_for_env(env)

        with contextlib.redirect_stdout(io.StringIO()):
            active_plan = solve_pddl(domain_path, problem_path)
        
        if not active_plan or len(active_plan.actions) == 0:
            break

        first_action = active_plan.actions[0]
        target_positions = extract_target_pos(first_action)

        if not target_positions:
            break

        active_agents = list(target_positions.keys())
        try:
            action_buffers = {
                agent: get_required_actions(env, agent, target_positions[agent])
                for agent in active_agents
            }
        except ValueError:
            obs, rewards, terms, truncs, _ = env.step({})
            steps_taken += 1
            if any(terms.values()) or any(truncs.values()):
                is_terminal = True
            continue

        longest_sequence = max(len(queue) for queue in action_buffers.values())
        for agent in active_agents:
            padding = [None] * (longest_sequence - len(action_buffers[agent]))
            action_buffers[agent] = padding + action_buffers[agent]

        while any(len(queue) > 0 for queue in action_buffers.values()):
            current_step_commands = {}
            for agent in active_agents:
                if action_buffers[agent]:
                    cmd = action_buffers[agent].pop(0)
                    if cmd is not None:
                        current_step_commands[agent] = cmd

            obs, rewards, terms, truncs, _ = env.step(current_step_commands)
            steps_taken += 1

            if any(terms.values()) or any(truncs.values()):
                is_terminal = True
                break

    return steps_taken

# ===========================================================================
# Part 2 - Modified Policy Iteration
# ===========================================================================

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
    env.reset()
    physics = BoxPushPhysicsEngine(env)
    
    start_state = physics.extract_start_state(env)
    queue = deque([start_state])
    visited = set([start_state])
    mdp_graph = {}
    
    all_joint_actions = list(itertools.product(range(4), repeat=2))
    
    while queue:
        current = queue.popleft()
        mdp_graph[current] = {}
        
        if len(mdp_graph) % 1000 == 0:
            print(f"Processed {len(mdp_graph)} states | Queue size: {len(queue)}")
            
        for ja in all_joint_actions:
            successors = physics.get_successors(current, ja)
            mdp_graph[current][ja] = successors
            
            for prob, next_state, reward in successors:
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)
                    
    print(f"Model mapped! Total reachable states: {len(mdp_graph)}")
    return mdp_graph

def modified_policy_iteration(env, gamma: float = 0.95, k: int = 15, max_outer_iters: int = 500):
    print("Mapping environment dynamics...")
    graph = build_transition_model(env)
    
    state_space = list(graph.keys())
    value_table = {s: 0.0 for s in state_space}
    policy_table = {s: next(iter(graph[s].keys())) for s in state_space}
    
    for iteration in range(max_outer_iters):
        final_delta = 0.0
        final_sweep_count = 0
        
        for sweep in range(k):
            max_value_shift = 0.0
            updated_values = {}
            for s in state_space:
                expected_v = 0.0
                for prob, n_state, rew in graph[s][policy_table[s]]:
                    expected_v += prob * (rew + gamma * value_table[n_state])
                
                max_value_shift = max(max_value_shift, abs(expected_v - value_table[s]))
                updated_values[s] = expected_v
            
            value_table = updated_values
            final_delta = max_value_shift
            final_sweep_count = sweep + 1
            if max_value_shift < 1e-5:
                break 
                
        print(f"  Iteration {iteration + 1:02d} | Sweeps utilized: {final_sweep_count}/{k} | Max Delta: {final_delta:.8f}")

        is_stable = True
        for s in state_space:
            current_choice = policy_table[s]
            top_choice, top_score = None, -float('inf')
            
            for candidate_act in graph[s].keys():
                score = sum(prob * (rew + gamma * value_table[n_state]) for prob, n_state, rew in graph[s][candidate_act])
                if score > top_score + 1e-8:
                    top_score = score
                    top_choice = candidate_act
                    
            policy_table[s] = top_choice
            if current_choice != top_choice:
                is_stable = False
                
        if is_stable:
            print(f"\nOptimal policy secured in {iteration + 1} macro-iterations.")
            break
            
    return policy_table, value_table

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
# Main - run both algorithms and print results
# ===========================================================================

if __name__ == "__main__":
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    mean_ol = 0.0
    std_ol = 0.0

    # ── Part 1: Online Planning ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 - Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    online_steps = []
    for i in range(100):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 10 == 0:
            print(f"  run {i+1}/100 - steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  →  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 - Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)

    # Initialize a single physics engine instance to use for state extraction
    physics_shim = BoxPushPhysicsEngine(env_mpi)

    def mpi_policy_fn(env, obs):
        # 1. Extract the current state using the exact same logic as the BFS graph
        state = physics_shim.extract_start_state(env)
        
        # Fallback: if we somehow slip into an unmapped state
        if state not in policy:
            return {env.possible_agents[0]: 0, env.possible_agents[1]: 0}
            
        joint_action = policy[state] 
        
        # 2. Map the abstract policy back to the physical agents
        # Since the abstract state sorted the agents by their (x, y) coordinates, 
        # we must sort the physical agent IDs by those exact same coordinates!
        ordered_roster = sorted(env.possible_agents, key=lambda a: tuple(env.agent_positions[a]))
        
        commands = {}
        for idx, agent in enumerate(ordered_roster):
            desired_heading = joint_action[idx]
            current_heading = env.agent_dirs[agent]
            
            # 3. Dynamic Rotation: Calculate how to turn if we aren't facing the target direction
            if current_heading == desired_heading:
                commands[agent] = 2  # 2 is Forward
            else:
                rot_diff = (desired_heading - current_heading) % 4
                # If target is 90 degrees right (1) or 180 degrees behind (2), turn Right (1)
                # If target is 90 degrees left (3), turn Left (0)
                commands[agent] = 1 if rot_diff in [1, 2] else 0 
                
        return commands

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