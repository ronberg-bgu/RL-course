import pygame
import time
import re

from planner.pddl_solver import solve_pddl
from environment.multi_agent_env import MultiAgentBoxPushEnv
from minigrid.core.constants import DIR_TO_VEC


# i have added new 2 methods (diff from visualize_plan.py)
# loc_to_xy and sim_xy_to_loc
def loc_to_xy(loc_name: str):
    parts = loc_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid location name: {loc_name}")

    row = int(parts[1])
    col = int(parts[2])

    x = col + 1
    y = row + 1
    return x, y

def sim_xy_to_loc(x, y):
    row = y - 1
    col = x - 1
    return f"loc_{row}_{col}"


# some changes diff from visualize_plan.py
def extract_target_pos(pddl_action):

    action_str = str(pddl_action)
    action_name = action_str.split("(")[0]
    params = re.findall(r"[\w_]+", action_str[action_str.find("(") + 1:action_str.find(")")])

    agent_targets = {}

    if action_name == "move" and len(params) >= 3:
        agent_name = params[0]
        target_loc = params[2]
        agent_targets[agent_name] = loc_to_xy(target_loc)

    elif action_name == "push-small" and len(params) >= 4:
        agent_name = params[0]
        box_loc = params[2]
        agent_targets[agent_name] = loc_to_xy(box_loc)

    elif action_name.startswith("push-big-") and len(params) >= 8:
        a1, a2 = params[0], params[1]
        box_loc = params[4]
        tgt = loc_to_xy(box_loc)
        agent_targets[a1] = tgt
        agent_targets[a2] = tgt

    return agent_targets


#unchanges from visualize_plan.py
def get_required_actions(env, agent, target_pos):
    """
    Returns a list of PettingZoo actions (0=left, 1=right, 2=forward)
    needed to move the agent into the target adjacent position.
    """
    current_pos = env.agent_positions[agent]
    current_dir = env.agent_dirs[agent]

    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    target_dir = None
    for d, vec in enumerate(DIR_TO_VEC):
        if vec[0] == dx and vec[1] == dy:
            target_dir = d
            break

    if target_dir is None:
        raise ValueError(f"Target {target_pos} is not adjacent to {current_pos}")

    actions = []
    while current_dir != target_dir:
        actions.append(1)  # 1 = turn right
        current_dir = (current_dir + 1) % 4
    actions.append(2)  # 2 = forward

    return actions

# some changes diff from visualize_plan.py
def execute_plan_in_env(plan, ascii_map, step_delay=0.2):
    env = MultiAgentBoxPushEnv(ascii_map=ascii_map, render_mode="human")
    env.reset()
    env.core_env.render()
    time.sleep(1)

    for pddl_action in plan.actions:
        print(f"\nExecuting: {pddl_action}")

        for a in env.agent_positions:
            x, y = env.agent_positions[a]
            print(f"  BEFORE {a}: {sim_xy_to_loc(int(x), int(y))}")

        agent_targets = extract_target_pos(pddl_action)
        if not agent_targets:
            print(f"Skipping unsupported action format: {pddl_action}")
            continue

        agents = list(agent_targets.keys())

        agent_action_queues = {}
        for a in agents:
            agent_action_queues[a] = get_required_actions(env, a, agent_targets[a])

        max_len = max(len(q) for q in agent_action_queues.values())
        for a in agents:
            agent_action_queues[a] = [None] * (max_len - len(agent_action_queues[a])) + agent_action_queues[a]

        while any(agent_action_queues[a] for a in agents):
            step_actions = {}

            for a in agents:
                if agent_action_queues[a]:
                    act = agent_action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act

            env.step(step_actions)
            env.core_env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            time.sleep(step_delay)

        for a in env.agent_positions:
            x, y = env.agent_positions[a]
            print(f"  AFTER  {a}: {sim_xy_to_loc(int(x), int(y))}")


    print("Plan execution complete.")
    time.sleep(3)
    pygame.quit()

def main():
    domain_file = "pddl_llm/domain.pddl"
    problem_file = "pddl_llm/problem.pddl"

    ascii_map = [
        "WWWWWWWW",
        "W  AA  W",
        "W B C  W",
        "W B    W",
        "W      W",
        "W      W",
        "W G GG W",
        "WWWWWWWW",
    ]

    print("Solving my LLM-generated PDDL files...- llm_pipline.py DEBUG")
    plan = solve_pddl(domain_file, problem_file)

    print("\nExecuting plan in simulator...- llm_pipline.py DEBUG")
    execute_plan_in_env(plan, ascii_map)


if __name__ == "__main__":
    main()