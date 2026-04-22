"""
llm_pipeline.py — Exercise 1: Multi-Agent Classical Planning with PettingZoo
=============================================================================
Pipeline (per exercises/README.md):
  1. Read the LLM-generated pddl/domain.pddl and pddl/problem.pddl.
  2. Call pddl_to_map to reconstruct the ASCII map from the PDDL files.
  3. Call the visualizer to solve and display the plan.
  4. Execute the plan step-by-step and print accumulated rewards.
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
PDDL_CREATION_PROMPT = '''
You are a PDDL expert. Generate two PDDL files — domain.pddl and problem.pddl — for the
multi-agent box-pushing planning problem described below.

═══════════════════════════════════════
  DOMAIN SPECIFICATION
═══════════════════════════════════════

Domain name: box-push

Requirements (exactly these, no others):
  (:requirements :strips :typing :equality)

Types (exactly these four):
  (:types agent location box heavybox)

Predicates (exactly these nine):
  (agent-at     ?a  - agent    ?loc - location)
  (box-at       ?b  - box      ?loc - location)
  (heavybox-at  ?b  - heavybox ?loc - location)
  (clear        ?loc - location)         ; true iff NO BOX is here — agents do NOT affect clear
  (adj          ?l1 - location ?l2 - location)
  (adj-left     ?l1 - location ?l2 - location)
  (adj-right    ?l1 - location ?l2 - location)
  (adj-up       ?l1 - location ?l2 - location)
  (adj-down     ?l1 - location ?l2 - location)

Actions — define all nine below. Read every constraint carefully.

─── move ───────────────────────────────
Parameters: ?a - agent  ?from - location  ?to - location
Precondition: (agent-at ?a ?from)  (adj ?from ?to)  (clear ?to)
Effect:       (agent-at ?a ?to)  (not (agent-at ?a ?from))
NOTE: NO (clear ?from) or (not (clear ?to)) effects. Agents never affect the clear predicate.

─── push-small-up / push-small-down / push-small-left / push-small-right ───
Parameters: ?a - agent  ?from - location  ?boxloc - location  ?toloc - location
            ?b - box  ?other - agent
Semantics:
  - ?a is the pushing agent at ?from
  - ?b is the small box currently at ?boxloc (one step ahead of ?a in the push direction)
  - ?toloc is where the box lands (one more step beyond ?boxloc in the same direction)
  - ?other is the OTHER agent (not the pusher) — required to block unsafe pushes
Precondition for push-small-up (mirror for the other three directions):
  (not (= ?a ?other))
  (agent-at ?a ?from)
  (adj-up ?from ?boxloc)
  (box-at ?b ?boxloc)
  (adj-up ?boxloc ?toloc)
  (clear ?toloc)
  (not (agent-at ?other ?toloc))   ← prevents the box landing on the other agent
Effect (same for all four directions):
  (agent-at ?a ?boxloc)       (not (agent-at ?a ?from))
  (box-at ?b ?toloc)          (not (box-at ?b ?boxloc))
  (not (clear ?toloc))
  (clear ?boxloc)
NOTE: NO (clear ?from) effect. The pusher leaving ?from does not make it clear
      (it was already clear — only boxes affect clear).

─── push-heavy-up / push-heavy-down / push-heavy-left / push-heavy-right ───
A heavy box requires TWO different agents standing on the SAME cell to push it.
Parameters: ?a1 - agent  ?a2 - agent  ?from - location  ?boxloc - location
            ?toloc - location  ?b - heavybox
Precondition for push-heavy-up (mirror for the other three directions):
  (not (= ?a1 ?a2))            ← CRITICAL — prevents planner reusing same agent as both pushers
  (agent-at ?a1 ?from)
  (agent-at ?a2 ?from)
  (adj-up ?from ?boxloc)
  (heavybox-at ?b ?boxloc)
  (adj-up ?boxloc ?toloc)
  (clear ?toloc)
Effect:
  (agent-at ?a1 ?boxloc)      (not (agent-at ?a1 ?from))
  (agent-at ?a2 ?boxloc)      (not (agent-at ?a2 ?from))
  (heavybox-at ?b ?toloc)     (not (heavybox-at ?b ?boxloc))
  (not (clear ?toloc))
  (clear ?boxloc)
NOTE: NO (clear ?from) effect. Two agents were at ?from; agents never affect clear.

═══════════════════════════════════════
  PROBLEM ENCODING RULES
═══════════════════════════════════════

1. LOCATION NAMING
   Every walkable cell (non-wall) gets a name: loc_X_Y
   X = column index (0 = leftmost), Y = row index (0 = topmost).
   Walls are simply absent — do not create location objects for wall cells.

2. ADJACENCY DIRECTION CONVENTION
   "up" means decreasing Y (moving toward the top of the map).
   (adj-up  loc_X_Y  loc_X_Y')  where Y' = Y - 1
   (adj-down loc_X_Y loc_X_Y')  where Y' = Y + 1
   (adj-left loc_X_Y loc_X_Y')  where X' = X - 1
   (adj-right loc_X_Y loc_X_Y') where X' = X + 1
   Also assert the undirected (adj ...) fact for every directional pair, in both directions:
     (adj loc_A loc_B)  and  (adj loc_B loc_A)
   Directional adj is one-way:
     (adj-up loc_X_Y loc_X_(Y-1)) — NOT the reverse

3. OBJECTS
   List every walkable location as "loc_X_Y - location".
   Name agents: agent_0, agent_1, ...
   Name small boxes: box_0, box_1, ...
   Name heavy boxes: hbx_0, hbx_1, ...

4. INIT STATE — include ALL of the following:
   a. One (agent-at agent_N loc_X_Y) per agent.
   b. One (box-at box_N loc_X_Y) per small box.
   c. One (heavybox-at hbx_N loc_X_Y) per heavy box.
   d. (clear loc_X_Y) for EVERY walkable location that does NOT currently have a box
      (small or heavy). This includes locations that have agents — agents do not block clear.
      A cell that starts with a box must NOT have a (clear ...) fact.
   e. Full adjacency facts — (adj ...) undirected and all four (adj-up/down/left/right ...)
      directed facts, one per adjacent walkable pair.

5. GOAL STATE
   Specify (box-at box_N loc_X_Y) and/or (heavybox-at hbx_N loc_X_Y) for every box
   that must reach a goal cell. Use explicit per-box goals — do NOT use (won) or
   any synthetic goal predicate.

═══════════════════════════════════════
  MAP TO ENCODE
═══════════════════════════════════════

Character key:
  W = wall (no location object)
  ' '(space) = empty walkable cell
  A = agent starting position (walkable, clear)
  B = small box starting position (walkable, NOT clear)
  C = heavy box starting position (1×1, walkable, NOT clear)
  G = goal cell (walkable, clear unless a box starts there)

[INSERT YOUR ASCII MAP HERE — one string per row, top to bottom]

═══════════════════════════════════════
  OUTPUT FORMAT
═══════════════════════════════════════

Produce two complete, syntactically valid PDDL files:

File 1 — domain.pddl
  All nine actions. No extra predicates, types, or actions beyond what is specified.

File 2 — problem.pddl
  All locations, agents, boxes, and heavy boxes as objects.
  Complete (:init ...) including ALL clear facts and ALL adjacency facts.
  (:goal (and ...)) using only box-at and heavybox-at predicates.

Do not abbreviate, omit adjacency facts, or leave any (clear ...) facts out.
Every missing clear fact or missing adjacency fact will cause the planner to fail.
'''
# ---------------------------------------------------------------------------
# PDDL files — located in exercises/ex1/pddl/
# ---------------------------------------------------------------------------
PDDL_DIR     = os.path.join(os.path.dirname(__file__), "pddl")
DOMAIN_PATH  = os.path.join(PDDL_DIR, "domain.pddl")
PROBLEM_PATH = os.path.join(PDDL_DIR, "problem.pddl")


# ---------------------------------------------------------------------------
# Action translation helpers
# ---------------------------------------------------------------------------

def extract_target_pos(pddl_action):
    """
    Parse a PDDL action string and return a dict mapping agent names to their
    target (x, y) grid coordinates for this step.

    Handles:
      move(?a ?from ?to)
          → agent enters ?to
      push-small-{up,down,left,right}(?a ?from ?boxloc ?toloc ?b)
          → agent enters ?boxloc
      push-heavy-{up,down,left,right}(?a1 ?a2 ?from ?boxloc ?toloc ?h)
          → both agents enter ?boxloc
    """
    action_str  = str(pddl_action)
    action_name = action_str.split("(")[0].strip()
    inner       = action_str[action_str.find("(") + 1: action_str.rfind(")")]
    params      = re.findall(r"[\w]+", inner)

    agent_targets = {}

    if action_name.startswith("win"):
        return {}

    if action_name.startswith("push-heavy") and len(params) >= 4:
        # Parameters: a1 a2 from boxloc toloc h
        a1, a2      = params[0], params[1]
        target_loc  = params[3]          # both agents step INTO boxloc
        parts = target_loc.split("_")
        if len(parts) == 3:
            tgt = (int(parts[1]), int(parts[2]))
            agent_targets[a1] = tgt
            agent_targets[a2] = tgt

    elif len(params) >= 3:
        # move:        a  from  to
        # push-small:  a  from  boxloc  toloc  b
        # In both cases the agent physically enters params[2].
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

    # --- Step 1: Reconstruct ASCII map from PDDL ----------------------------
    print("\n[1] Reconstructing ASCII map from PDDL files...")
    from pddl_to_map import parse_pddl_to_map
    ascii_map = parse_pddl_to_map(DOMAIN_PATH, PROBLEM_PATH)
    print(f"    Map size: {len(ascii_map)} rows x {len(ascii_map[0])} cols")
    for row in ascii_map:
        print(f"    {row}")

    # --- Step 2: Build environment -------------------------------------------
    print("\n[2] Building environment from reconstructed map...")
    env = MultiAgentBoxPushEnv(ascii_map=ascii_map, render_mode=None, max_steps=500)
    env.reset()
    print(f"    Agents : {env.possible_agents}")

    # --- Step 3: PDDL files --------------------------------------------------
    print(f"\n[3] PDDL files:")
    print(f"    domain  -> {DOMAIN_PATH}")
    print(f"    problem -> {PROBLEM_PATH}")

    # --- Step 4: Visualize the plan ------------------------------------------
    print("\n[4] Running visualization...")
    from visualize_plan import visualize_pddl_plan
    visualize_pddl_plan(ascii_map, DOMAIN_PATH, PROBLEM_PATH)

    # --- Step 5: Execute plan and collect rewards ----------------------------
    print("\n[5] Executing plan to collect rewards...")
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

        # Pad shorter queues with None at the front so all agents execute
        # their final 'forward' in the same env.step() — required for push-heavy.
        if queues:
            max_len = max(len(q) for q in queues.values())
            for a in queues:
                queues[a] = [None] * (max_len - len(queues[a])) + queues[a]

        while any(queues.values()):
            step_actions = {a: act for a, q in queues.items()
                            if q and (act := q.pop(0)) is not None}
            _, rewards, terminations, truncations, _ = env.step(step_actions)
            for a, r in rewards.items():
                total_rewards[a] += r
            if any(terminations.values()) or any(truncations.values()):
                break

        if not env.agents:
            break

    # --- Step 6: Print results -----------------------------------------------
    print("\n[6] Total accumulated rewards:")
    for agent, reward in total_rewards.items():
        print(f"    {agent}: {reward}")
    print(f"\n    Summary: {total_rewards}")


if __name__ == "__main__":
    main()
