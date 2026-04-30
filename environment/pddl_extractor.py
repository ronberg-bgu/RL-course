import os

def generate_domain(domain_path):
    domain_str = """(define (domain box-pushing)
          (:requirements :strips :typing :equality)

          (:types 
            agent 
            location 
            box 
            heavybox
            direction
          )

          (:constants
            dir-up dir-down dir-left dir-right - direction
          )

          (:predicates
            (agent-at ?a - agent ?l - location)
            (box-at ?b - box ?l - location)
            (heavybox-at ?b - heavybox ?l - location)

            (adj-dir ?from ?to - location ?d - direction)

            ;; only boxes affect this
            (clear ?l - location)
          )

          ;; MOVE
          (:action move
            :parameters (?a - agent ?from ?to - location ?d - direction)
            :precondition (and
              (agent-at ?a ?from)
              (adj-dir ?from ?to ?d)
              (clear ?to)
            )
            :effect (and
              (not (agent-at ?a ?from))
              (agent-at ?a ?to)
            )
          )

          ;; PUSH SMALL
          (:action push-small
            :parameters (?a - agent ?from ?boxloc ?toloc - location ?b - box ?d - direction)
            :precondition (and
              (agent-at ?a ?from)
              (box-at ?b ?boxloc)

              (adj-dir ?from ?boxloc ?d)
              (adj-dir ?boxloc ?toloc ?d)

              (clear ?toloc)
            )
            :effect (and
              (not (agent-at ?a ?from))
              (agent-at ?a ?boxloc)

              (not (box-at ?b ?boxloc))
              (box-at ?b ?toloc)

              (clear ?boxloc)
              (not (clear ?toloc))
            )
          )

          ;; PUSH HEAVY
          (:action push-heavy
            :parameters (?a1 ?a2 - agent ?from ?boxloc ?toloc - location ?b - heavybox ?d - direction)
            :precondition (and
              (not (= ?a1 ?a2))
              (agent-at ?a1 ?from)
              (agent-at ?a2 ?from)

              (heavybox-at ?b ?boxloc)

              (adj-dir ?from ?boxloc ?d)
              (adj-dir ?boxloc ?toloc ?d)

              (clear ?toloc)
            )
            :effect (and
              (not (agent-at ?a1 ?from))
              (not (agent-at ?a2 ?from))
              (agent-at ?a1 ?boxloc)
              (agent-at ?a2 ?boxloc)

              (not (heavybox-at ?b ?boxloc))
              (heavybox-at ?b ?toloc)

              (clear ?boxloc)
              (not (clear ?toloc))
            )
          )
        )
    """
    with open(domain_path, "w") as f:
        f.write(domain_str)

def generate_problem(env, problem_path):
    """
    Updated PDDL generator compatible with directional domain (adj-dir).
    Goals come from env.goal_positions (static).
    """

    w, h = env.width, env.height

    locations = []
    adjacencies = []
    agents = env.agents
    boxes = []
    heavyboxes = []

    # ── GRID SCAN ───────────────────────────────────────────────
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)

            is_wall = (cell is not None and cell.type == 'wall')
            if not is_wall:
                loc = f"loc_{x}_{y}"
                locations.append(loc)

                # RIGHT / LEFT
                if x < w - 1:
                    r_cell = env.core_env.grid.get(x + 1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x+1}_{y}", "dir-right"))
                        adjacencies.append((f"loc_{x+1}_{y}", loc, "dir-left"))

                # DOWN / UP
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y + 1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x}_{y+1}", "dir-up"))
                        adjacencies.append((f"loc_{x}_{y+1}", loc, "dir-down"))

                # BOXES
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        heavyboxes.append((f"hbox_{len(heavyboxes)}", loc))
                    else:
                        boxes.append((f"box_{len(boxes)}", loc))

    # ── GOALS (STATIC) ──────────────────────────────────────────
    goals = [f"loc_{gx}_{gy}" for gx, gy in env.goal_positions]

    # ── AGENTS ─────────────────────────────────────────────────
    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))

    # ── CLEAR (ONLY BOXES MATTER!) ─────────────────────────────
    clear_set = set(locations)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, loc in heavyboxes:
        clear_set.discard(loc)

    # ── OBJECTS ────────────────────────────────────────────────
    obj_str = "    " + " ".join(locations) + " - location\n"

    if agents:
        obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes:
        obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if heavyboxes:
        obj_str += "    " + " ".join([b[0] for b in heavyboxes]) + " - heavybox\n"

    # ── INIT ───────────────────────────────────────────────────
    init_str = ""

    for loc in clear_set:
        init_str += f"    (clear {loc})\n"

    for a, loc in agent_locs:
        init_str += f"    (agent-at {a} {loc})\n"

    for b, loc in boxes:
        init_str += f"    (box-at {b} {loc})\n"

    for hbox, loc in heavyboxes:
        init_str += f"    (heavybox-at {hbox} {loc})\n"

    for l1, l2, d in adjacencies:
        init_str += f"    (adj-dir {l1} {l2} {d})\n"

    # ── GOAL ───────────────────────────────────────────────────
    goal_conditions = []

    for i, (b_name, _) in enumerate(boxes):
        if i < len(goals):
            goal_conditions.append(f"(box-at {b_name} {goals[i]})")

    for j, (h_name, _) in enumerate(heavyboxes):
        idx = len(boxes) + j
        if idx < len(goals):
            goal_conditions.append(f"(heavybox-at {h_name} {goals[idx]})")

    if goal_conditions:
        goal_str = "(and\n" + "\n".join(f"    {g}" for g in goal_conditions) + "\n  )"
    else:
        goal_str = "(and)"

    # ── FINAL PDDL ─────────────────────────────────────────────
    problem_str = f"""(define (problem bp-map)
  (:domain box-pushing)
  (:objects
{obj_str}  )
  (:init
{init_str}  )
  (:goal
    {goal_str}
  )
)
"""

    with open(problem_path, "w") as f:
        f.write(problem_str)


def generate_pddl_for_env(env, pddl_folder="pddl"):
    os.makedirs(pddl_folder, exist_ok=True)
    domain_path = os.path.join(pddl_folder, "domain.pddl")
    problem_path = os.path.join(pddl_folder, "problem.pddl")
    
    generate_domain(domain_path)
    generate_problem(env, problem_path)
    
    return domain_path, problem_path
