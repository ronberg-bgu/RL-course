import os


DIRECTIONS = {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}


def generate_domain(domain_path):
    domain_str = """(define (domain box-push)
  (:requirements :strips :typing :equality)
  (:types agent location box heavybox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (right-of ?from - location ?to - location)
    (left-of ?from - location ?to - location)
    (down-of ?from - location ?to - location)
    (up-of ?from - location ?to - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and
      (agent-at ?a ?from)
      (adj ?from ?to)
    )
    :effect (and
      (agent-at ?a ?to)
      (not (agent-at ?a ?from))
      (clear ?from)
    )
  )

  (:action push-small-right
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (right-of ?from ?boxloc)
      (right-of ?boxloc ?toloc)
      (box-at ?b ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc)
      (not (agent-at ?a ?from))
      (clear ?from)
      (box-at ?b ?toloc)
      (not (box-at ?b ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-small-left
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (left-of ?from ?boxloc)
      (left-of ?boxloc ?toloc)
      (box-at ?b ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc)
      (not (agent-at ?a ?from))
      (clear ?from)
      (box-at ?b ?toloc)
      (not (box-at ?b ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-small-down
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (down-of ?from ?boxloc)
      (down-of ?boxloc ?toloc)
      (box-at ?b ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc)
      (not (agent-at ?a ?from))
      (clear ?from)
      (box-at ?b ?toloc)
      (not (box-at ?b ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-small-up
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (up-of ?from ?boxloc)
      (up-of ?boxloc ?toloc)
      (box-at ?b ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc)
      (not (agent-at ?a ?from))
      (clear ?from)
      (box-at ?b ?toloc)
      (not (box-at ?b ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-heavy-right
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (right-of ?from ?boxloc)
      (right-of ?boxloc ?toloc)
      (heavybox-at ?h ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (clear ?from)
      (heavybox-at ?h ?toloc)
      (not (heavybox-at ?h ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-heavy-left
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (left-of ?from ?boxloc)
      (left-of ?boxloc ?toloc)
      (heavybox-at ?h ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (clear ?from)
      (heavybox-at ?h ?toloc)
      (not (heavybox-at ?h ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-heavy-down
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (down-of ?from ?boxloc)
      (down-of ?boxloc ?toloc)
      (heavybox-at ?h ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (clear ?from)
      (heavybox-at ?h ?toloc)
      (not (heavybox-at ?h ?boxloc))
      (not (clear ?toloc))
    )
  )

  (:action push-heavy-up
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (up-of ?from ?boxloc)
      (up-of ?boxloc ?toloc)
      (heavybox-at ?h ?boxloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (clear ?from)
      (heavybox-at ?h ?toloc)
      (not (heavybox-at ?h ?boxloc))
      (not (clear ?toloc))
    )
  )
)
"""
    with open(domain_path, "w") as f:
        f.write(domain_str)


def loc_name(x, y):
    return f"loc_{x}_{y}"


def is_wall(env, x, y):
    cell = env.core_env.grid.get(x, y)
    return cell is not None and cell.type == "wall"


def generate_problem(env, problem_path):
    w, h = env.width, env.height

    locations = []
    adjacencies = []
    directional_adjacencies = {
        "right": [],
        "left": [],
        "down": [],
        "up": [],
    }

    agents = env.agents
    boxes = []
    heavyboxes = []

    goals = []
    for y, row in enumerate(env.ascii_map):
        for x, char in enumerate(row):
            if char == "G":
                goals.append(loc_name(x, y))

    for y in range(h):
        for x in range(w):
            if is_wall(env, x, y):
                continue

            loc = loc_name(x, y)
            locations.append(loc)

            for direction_name, (dx, dy) in DIRECTIONS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not is_wall(env, nx, ny):
                    nloc = loc_name(nx, ny)
                    adjacencies.append((loc, nloc))
                    directional_adjacencies[direction_name].append((loc, nloc))

            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavyboxes.append((f"hbx_{len(heavyboxes)}", loc))
                else:
                    boxes.append((f"box_{len(boxes)}", loc))

    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, loc_name(px, py)))

    clear_set = set(locations)

    for _, loc in agent_locs:
        clear_set.discard(loc)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, loc in heavyboxes:
        clear_set.discard(loc)

    obj_str = "    " + " ".join(locations) + " - location\n"

    if agents:
        obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes:
        obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if heavyboxes:
        obj_str += "    " + " ".join([b[0] for b in heavyboxes]) + " - heavybox\n"

    init_str = ""

    for loc in sorted(clear_set):
        init_str += f"    (clear {loc})\n"

    for a, loc in agent_locs:
        init_str += f"    (agent-at {a} {loc})\n"

    for b, loc in boxes:
        init_str += f"    (box-at {b} {loc})\n"

    for hb, loc in heavyboxes:
        init_str += f"    (heavybox-at {hb} {loc})\n"

    for l1, l2 in adjacencies:
        init_str += f"    (adj {l1} {l2})\n"

    for l1, l2 in directional_adjacencies["right"]:
        init_str += f"    (right-of {l1} {l2})\n"
    for l1, l2 in directional_adjacencies["left"]:
        init_str += f"    (left-of {l1} {l2})\n"
    for l1, l2 in directional_adjacencies["down"]:
        init_str += f"    (down-of {l1} {l2})\n"
    for l1, l2 in directional_adjacencies["up"]:
        init_str += f"    (up-of {l1} {l2})\n"

    goal_conditions = []

    for i, (b_name, _) in enumerate(boxes):
        if i < len(goals):
            goal_conditions.append(f"(box-at {b_name} {goals[i]})")

    for j, (h_name, _) in enumerate(heavyboxes):
        goal_idx = len(boxes) + j
        if goal_idx < len(goals):
            goal_conditions.append(f"(heavybox-at {h_name} {goals[goal_idx]})")

    if goal_conditions:
        goal_str = "(and\n" + "\n".join(f"    {g}" for g in goal_conditions) + "\n  )"
    else:
        goal_str = "(and)"

    problem_str = f"""(define (problem bp-map)
  (:domain box-push)
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