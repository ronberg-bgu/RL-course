import random
import re
import statistics
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, List, Optional, Tuple

import unified_planning.shortcuts as up
from minigrid.core.constants import DIR_TO_VEC

from environment.multi_agent_env import MultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl


up.get_environment().credits_stream = None


DEFAULT_MAP = [
    "WWWWWWWW",
    "W AA   W",
    "W  B B W",
    "W  G G W",
    "W   C  W",
    "W   G  W",
    "WWWWWWWW",
]


def parse_pddl_action(pddl_action) -> Tuple[str, List[str]]:
    action_str = str(pddl_action).strip()
    action_name = action_str.split("(")[0]
    params_str = action_str[action_str.find("(") + 1 : action_str.rfind(")")]
    params = re.findall(r"[\w_]+", params_str)
    return action_name, params


def loc_to_xy(loc_name: str) -> Tuple[int, int]:
    parts = loc_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad location format: {loc_name}")
    return int(parts[1]), int(parts[2])


def direction_from_to(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]

    for d, vec in enumerate(DIR_TO_VEC):
        if int(vec[0]) == dx and int(vec[1]) == dy:
            return d

    raise ValueError(f"{dst} is not adjacent to {src}")


def rotation_actions(current_dir: int, target_dir: int) -> List[int]:
    right_steps = (target_dir - current_dir) % 4
    left_steps = (current_dir - target_dir) % 4

    if right_steps <= left_steps:
        return [1] * right_steps
    return [0] * left_steps


def env_step(env, actions: Dict[str, int]):
    return env.step(actions)


def align_agent_to_direction(env, agent: str, target_dir: int):
    current_dir = env.agent_dirs[agent]
    for act in rotation_actions(current_dir, target_dir):
        env_step(env, {agent: act})


def align_agents_same_direction(env, agents: List[str], target_dir: int):
    pending = {}

    for agent in agents:
        current_dir = env.agent_dirs[agent]
        pending[agent] = rotation_actions(current_dir, target_dir)

    while any(queue for queue in pending.values()):
        joint_action = {}
        for agent, queue in pending.items():
            if queue:
                joint_action[agent] = queue.pop(0)
        env_step(env, joint_action)


def sideways_dirs(move_dir: int) -> Tuple[int, int]:
    return (move_dir - 1) % 4, (move_dir + 1) % 4


def stochastic_move_direction(intended_dir: int) -> int:
    r = random.random()
    left_dir, right_dir = sideways_dirs(intended_dir)

    if r < 0.8:
        return intended_dir
    if r < 0.9:
        return left_dir
    return right_dir


def execute_stochastic_move(env, agent: str, intended_target: Tuple[int, int]):
    current_pos = env.agent_positions[agent]
    intended_dir = direction_from_to(current_pos, intended_target)

    actual_dir = stochastic_move_direction(intended_dir)

    align_agent_to_direction(env, agent, actual_dir)
    env_step(env, {agent: 2})


def execute_stochastic_push_small(
    env,
    agent: str,
    from_loc: Tuple[int, int],
    box_loc: Tuple[int, int],
):
    push_dir = direction_from_to(from_loc, box_loc)
    align_agent_to_direction(env, agent, push_dir)

    if random.random() < 0.8:
        env_step(env, {agent: 2})


def execute_stochastic_push_heavy(
    env,
    a1: str,
    a2: str,
    from_loc: Tuple[int, int],
    box_loc: Tuple[int, int],
):
    push_dir = direction_from_to(from_loc, box_loc)
    align_agents_same_direction(env, [a1, a2], push_dir)

    if random.random() < 0.8:
        env_step(env, {a1: 2, a2: 2})


def execute_first_planner_action(env, planner_action) -> bool:
    action_name, params = parse_pddl_action(planner_action)

    if action_name == "move":
        agent = params[0]
        target = loc_to_xy(params[2])
        execute_stochastic_move(env, agent, target)
        return True

    if action_name.startswith("push-small"):
        agent = params[0]
        from_loc = loc_to_xy(params[1])
        box_loc = loc_to_xy(params[2])
        execute_stochastic_push_small(env, agent, from_loc, box_loc)
        return True

    if action_name.startswith("push-heavy"):
        a1 = params[0]
        a2 = params[1]
        from_loc = loc_to_xy(params[2])
        box_loc = loc_to_xy(params[3])
        execute_stochastic_push_heavy(env, a1, a2, from_loc, box_loc)
        return True

    if action_name.startswith("win"):
        return True

    print(f"Unrecognized planner action: {planner_action}")
    return False


def solve_pddl_quiet(domain_path: str, problem_path: str):
    buffer = StringIO()
    with redirect_stdout(buffer):
        return solve_pddl(domain_path, problem_path)


def run_online_replanning_episode(
    ascii_map: List[str],
    max_steps: int = 100,
    render: bool = False,
    seed: Optional[int] = None,
) -> Tuple[bool, int]:
    if seed is not None:
        random.seed(seed)

    env = MultiAgentBoxPushEnv(
        ascii_map=ascii_map,
        render_mode="human" if render else None,
        max_steps=max_steps,
    )

    env.reset(seed=seed)

    steps = 0
    same_plan_counter = 0
    last_first_action = None

    while steps < max_steps:
        domain_path, problem_path = generate_pddl_for_env(env, pddl_folder="pddl")
        plan = solve_pddl_quiet(domain_path, problem_path)

        if plan is None or not getattr(plan, "actions", None):
            print("No plan found.")
            return False, steps

        first_action = str(plan.actions[0])
        print(f"[Step {steps}] First action: {first_action}")

        if first_action == last_first_action:
            same_plan_counter += 1
        else:
            same_plan_counter = 0

        last_first_action = first_action

        if same_plan_counter >= 15:
            print("Same first action repeated too many times -> stopping.")
            return False, steps

        ok = execute_first_planner_action(env, plan.actions[0])
        if not ok:
            return False, steps

        steps += 1

        if len(env.agents) == 0:
            print("Environment terminated.")
            return True, steps

    print("Max steps reached.")
    return False, steps


def evaluate_online_replanning(
    ascii_map: List[str],
    n_runs: int = 100,
    max_steps: int = 100,
) -> Dict[str, float]:
    successful_steps = []
    successes = 0

    for run_idx in range(n_runs):
        success, steps = run_online_replanning_episode(
            ascii_map=ascii_map,
            max_steps=max_steps,
            render=False,
            seed=run_idx,
        )

        if success:
            successes += 1
            successful_steps.append(steps)

    return {
        "runs": n_runs,
        "successes": successes,
        "success_rate": successes / n_runs if n_runs else 0.0,
        "mean_steps": statistics.mean(successful_steps) if successful_steps else float("inf"),
        "std_steps": statistics.pstdev(successful_steps) if len(successful_steps) > 1 else 0.0,
    }


def main():
    print("Running one debug episode...")

    success, steps = run_online_replanning_episode(
        ascii_map=DEFAULT_MAP,
        max_steps=100,
        render=False,
        seed=0,
    )

    print(f"Single run -> success={success}, steps={steps}")

    print("\nRunning small evaluation test...")
    stats = evaluate_online_replanning(
        ascii_map=DEFAULT_MAP,
        n_runs=3,
        max_steps=100,
    )

    print(stats)


if __name__ == "__main__":
    main()