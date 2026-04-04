"""
hw1_solution.py — Exercise 1: Multi-Agent Classical Planning with PettingZoo
=============================================================================
Task requirements (from exercises/README.md):
  1. Define a custom map: ≥3 corridor walls, 2 agents, 1 BigBox, 2 SmallBoxes.
  2. Use domain.pddl + problem.pddl for the PDDL representation.
  3. Call solve_pddl() to compute an optimal logical plan.
  4. Execute the plan step-by-step in the PettingZoo environment (no visualizer).
  5. Print the total accumulated rewards dict after the maze is solved.
"""

import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Path setup — run from the repo root or from exercises/ex1/
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from minigrid.core.constants import DIR_TO_VEC
from environment.multi_agent_env import MultiAgentBoxPushEnv

# ---------------------------------------------------------------------------
# Custom map
# ---------------------------------------------------------------------------
# Layout (0-indexed, x=col, y=row), grid is 10 wide × 9 tall:
#
#   Row 0: WWWWWWWWWW
#   Row 1: W   W    W   ← wall corridor at col 4
#   Row 2: W A W B  W   ← agent_0 at (2,2), SmallBox at (6,2), wall at col 4
#   Row 3: W      W W   ← wall at col 7
#   Row 4: W  B CCW W   ← SmallBox at (3,4), BigBox at (5,4)+(6,4), wall at col 7
#   Row 5: W W      W   ← wall at col 2
#   Row 6: W W A GG W   ← agent_1 at (4,6), Goals at (6,6)+(7,6), wall at col 2
#   Row 7: W    GG  W   ← Goals at (5,7)+(6,7) for BigBox
#   Row 8: WWWWWWWWWW
#
# Complexity:
#   • 6 internal wall tiles forming 3 corridor walls (col 4 rows 1-2,
#     col 7 rows 3-4, col 2 rows 5-6)
#   • 2 agents (agent_0, agent_1)
#   • 1 BigBox spanning (5,4)+(6,4) — needs cooperative push
#   • 2 SmallBoxes at (6,2) and (3,4) — each pushable by a single agent
#   • Goals: SmallBox goals at (6,6) and (7,6); BigBox goal at (5,7)+(6,7)

ex1_map = [
    "WWWWWWWWWW",
    "W   W    W",
    "W A W B  W",
    "W      W W",
    "W  B CCW W",
    "W W      W",
    "W W A GG W",
    "W    GG  W",
    "WWWWWWWWWW",
]

# ---------------------------------------------------------------------------
# PDDL files — located in exercises/ex1/pddl/
# ---------------------------------------------------------------------------
PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")
DOMAIN_PATH = os.path.join(PDDL_DIR, "domain.pddl")
PROBLEM_PATH = os.path.join(PDDL_DIR, "problem.pddl")


# ---------------------------------------------------------------------------
# Action translation helpers
# ---------------------------------------------------------------------------

def extract_target_pos(pddl_action):
    """
    Parse a PDDL action string and return a dict mapping agent names to their
    target (x, y) grid coordinates for this step.

    Handles:
      move(?a ?from ?to)                              → agent enters ?to
      push-small-{up,down,left,right}(?a ?from ?boxloc ?toloc ?b)
                                                      → agent enters ?boxloc
      push-big-{up,down,left,right}(?a1 ?a2 ?from1 ?from2 ?boxloc1 ?boxloc2 ...)
                                                      → agents enter ?boxloc1 / ?boxloc2
    """
    action_str = str(pddl_action)
    action_name = action_str.split("(")[0].strip()
    inner = action_str[action_str.find("(") + 1 : action_str.rfind(")")]
    params = re.findall(r"[\w]+", inner)

    agent_targets = {}

    if action_name.startswith("win"):
        return {}

    if action_name.startswith("push-big") and len(params) >= 6:
        # Parameters: a1 a2 from1 from2 boxloc1 boxloc2 toloc1 toloc2 b
        a1, a2 = params[0], params[1]
        t1, t2 = params[4], params[5]   # agents step INTO boxloc1 / boxloc2
        for agent, tok in ((a1, t1), (a2, t2)):
            parts = tok.split("_")
            if len(parts) == 3:
                agent_targets[agent] = (int(parts[1]), int(parts[2]))
    elif len(params) >= 3:
        # move: a from to
        # push-small-{up,down,left,right}: a from boxloc toloc b
        # In both cases the agent physically enters params[2]
        agent_name = params[0]
        target_loc = params[2]
        parts = target_loc.split("_")
        if len(parts) == 3:
            agent_targets[agent_name] = (int(parts[1]), int(parts[2]))

    return agent_targets


def get_pz_actions(env, agent, target_pos):
    """
    Return a list of PettingZoo integer actions (0=turn-left, 1=turn-right,
    2=forward) that turn the agent to face target_pos, then move forward.
    """
    current_pos = env.agent_positions[agent]
    current_dir = env.agent_dirs[agent]

    dx = int(target_pos[0]) - int(current_pos[0])
    dy = int(target_pos[1]) - int(current_pos[1])

    target_dir = None
    for d, vec in enumerate(DIR_TO_VEC):
        if vec[0] == dx and vec[1] == dy:
            target_dir = d
            break

    if target_dir is None:
        raise ValueError(
            f"Agent '{agent}' at {current_pos}: "
            f"target {target_pos} is not adjacent."
        )

    actions = []
    d = current_dir
    while d != target_dir:
        actions.append(1) 
        d = (d + 1) % 4
    actions.append(2) 
    return actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  HW1 – Multi-Agent Classical Planning Solution")
    print("=" * 60)

    # --- Step 1: Build environment -------------------------------------------
    print("\n[1] Building environment from custom map...")
    env = MultiAgentBoxPushEnv(ascii_map=ex1_map, render_mode=None, max_steps=500)
    env.reset()
    print(f"    Agents : {env.possible_agents}")
    print(f"    Map    : {len(ex1_map)} rows x {len(ex1_map[0])} cols")

    # --- Step 2: PDDL files --------------------------------------------------
    print(f"\n[2] PDDL files:")
    print(f"    domain  -> {DOMAIN_PATH}")
    print(f"    problem -> {PROBLEM_PATH}")

    # --- Step 3: Visualize the plan ------------------------------------------
    print("\n[3] Running visualization...")
    from visualize_plan import visualize_pddl_plan
    visualize_pddl_plan(ex1_map, DOMAIN_PATH, PROBLEM_PATH)

    # --- Step 4: Execute plan and collect rewards ----------------------------
    print("\n[4] Executing plan to collect rewards...")
    from planner.pddl_solver import solve_pddl
    plan = solve_pddl(DOMAIN_PATH, PROBLEM_PATH)
    if plan is None:
        print("No plan found. Exiting.")
        return

    env.reset()
    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    for pddl_action in plan.actions:
        agent_targets = extract_target_pos(pddl_action)
        if not agent_targets:
            continue

        queues = {a: get_pz_actions(env, a, t) for a, t in agent_targets.items()}

        while any(queues.values()):
            step_actions = {a: q.pop(0) for a, q in queues.items() if q}
            _, rewards, terminations, truncations, _ = env.step(step_actions)
            for a, r in rewards.items():
                total_rewards[a] += r
            if any(terminations.values()) or any(truncations.values()):
                break

        if not env.agents:
            break

    # --- Step 5: Print results -----------------------------------------------
    print("\n[5] Total accumulated rewards:")
    for agent, reward in total_rewards.items():
        print(f"    {agent}: {reward}")
    print(f"\n    Summary: {total_rewards}")


if __name__ == "__main__":
    main()
