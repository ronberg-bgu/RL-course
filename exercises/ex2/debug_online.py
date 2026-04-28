"""Run ONE episode with debug prints to verify the loop-breaker works."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions
import unified_planning as up
up.shortcuts.get_environment().credits_stream = None

ASCII_MAP = [
    "WWWWWWWW",
    "W  AA  W",
    "W B C  W",
    "W      W",
    "W   B  W",
    "W G G GW",
    "WWWWWWWW",
]


def run_one_episode_debug(env, max_replans=500):
    obs, _ = env.reset()
    total_env_steps = 0
    done = False
    last_action_str = None
    stuck_count = 0
    perturbations = 0

    for iter_num in range(max_replans):
        if done:
            print(f"\n>>> GOAL REACHED in {total_env_steps} env-steps, {iter_num} iterations, {perturbations} perturbations")
            return total_env_steps

        # Show current agent + box positions
        a0 = tuple(env.agent_positions[env.possible_agents[0]])
        a1 = tuple(env.agent_positions[env.possible_agents[1]])
        boxes = []
        for y in range(env.height):
            for x in range(env.width):
                c = env.core_env.grid.get(x, y)
                if c is not None and c.type == "box":
                    kind = getattr(c, "box_size", "small")
                    boxes.append(f"{kind}@{(x,y)}")

        domain_path, problem_path = generate_pddl_for_env(env)
        plan = solve_pddl(domain_path, problem_path)
        if not plan or len(plan.actions) == 0:
            print(f"\n>>> PLANNER RETURNED EMPTY PLAN at iter {iter_num} — goal reached?")
            return total_env_steps

        pddl_action = plan.actions[0]
        action_str = str(pddl_action)

        if action_str == last_action_str:
            stuck_count += 1
        else:
            stuck_count = 0
        last_action_str = action_str

        print(f"\n[iter {iter_num}] env_steps={total_env_steps} a0={a0} a1={a1} boxes={boxes}")
        print(f"    plan[0] = {action_str}")
        print(f"    stuck_count = {stuck_count}")

        if stuck_count >= 3:
            perturbations += 1
            step_actions = {a: random.choice([0, 1, 2]) for a in env.possible_agents}
            print(f"    *** STUCK — PERTURBING with {step_actions} (perturbation #{perturbations})")
            obs, _, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
            stuck_count = 0
            last_action_str = None
            continue

        agent_targets = extract_target_pos(pddl_action)
        if not agent_targets:
            print(f"    (no agent_targets — breaking)")
            break

        agents_in_action = list(agent_targets.keys())
        try:
            action_queues = {
                a: get_required_actions(env, a, agent_targets[a])
                for a in agents_in_action
            }
        except ValueError as e:
            print(f"    get_required_actions raised: {e} — bumping stuck_count")
            stuck_count += 3
            continue

        max_len = max((len(q) for q in action_queues.values() if q is not None), default=0)
        for a in agents_in_action:
            if action_queues[a] is None:
                action_queues[a] = []
            action_queues[a] = [None] * (max_len - len(action_queues[a])) + action_queues[a]

        queue_len_before = sum(len(q) for q in action_queues.values())
        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {agent: 4 for agent in env.possible_agents}
            for a in agents_in_action:
                if action_queues[a]:
                    act = action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act
            obs, _, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1
            if any(terms.values()) or any(truncs.values()):
                done = True
                break

        a0_after = tuple(env.agent_positions[env.possible_agents[0]])
        print(f"    after queue ({queue_len_before} cmds): a0={a0_after} done={done}")

    print(f"\n>>> HIT MAX_REPLANS ({max_replans}) — failed. total_env_steps={total_env_steps}, perturbations={perturbations}")
    return total_env_steps


if __name__ == "__main__":
    random.seed(42)  # reproducible
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=2000)
    steps = run_one_episode_debug(env, max_replans=500)
    print(f"\nFINAL: {steps} env-steps")