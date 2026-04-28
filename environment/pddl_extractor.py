import os

# Stable mapping for box objects to names
_box_to_name = {}

def get_box_name(cell, box_type):
    global _box_to_name
    cell_id = id(cell)
    if cell_id not in _box_to_name:
        prefix = "hbx" if box_type == "heavy" else "box"
        _box_to_name[cell_id] = f"{prefix}_{len(_box_to_name)}"
    return _box_to_name[cell_id]

def generate_domain(domain_path):
    domain_str = """(define (domain box-push)
  (:requirements :strips :typing :equality)
  (:types agent location box heavybox direction)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (move-dir ?l1 - location ?l2 - location ?d - direction)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?d - direction)
    :precondition (and (agent-at ?a ?from) (move-dir ?from ?boxloc ?d) (box-at ?b ?boxloc) (move-dir ?boxloc ?toloc ?d) (clear ?toloc))
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)) (clear ?boxloc))
  )

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox ?d - direction)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from)
        (agent-at ?a2 ?from)
        (move-dir ?from ?boxloc ?d)
        (heavybox-at ?h ?boxloc)
        (move-dir ?boxloc ?toloc ?d)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a1 ?boxloc)
        (agent-at ?a2 ?boxloc)
        (not (agent-at ?a1 ?from))
        (not (agent-at ?a2 ?from))
        (heavybox-at ?h ?toloc)
        (not (heavybox-at ?h ?boxloc))
        (not (clear ?toloc))
        (clear ?boxloc)
    )
  )
)
"""
    with open(domain_path, "w") as f:
        f.write(domain_str)

def generate_problem(env, problem_path):
    w, h = env.width, env.height
    locations = []
    adjacencies = []
    agents = env.agents
    boxes = []
    heavyboxes = []

    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is None or cell.type != 'wall':
                loc = f"loc_{x}_{y}"
                locations.append(loc)
                if x < w - 1:
                    r_cell = env.core_env.grid.get(x+1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x+1}_{y}", "right"))
                        adjacencies.append((f"loc_{x+1}_{y}", loc, "left"))
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y+1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x}_{y+1}", "down"))
                        adjacencies.append((f"loc_{x}_{y+1}", loc, "up"))

                if cell is not None and cell.type == "box":
                    b_type = "heavy" if getattr(cell, "box_size", "") == "heavy" else "small"
                    b_name = get_box_name(cell, b_type)
                    if b_type == "heavy": heavyboxes.append((b_name, loc))
                    else: boxes.append((b_name, loc))

    goals = [f"loc_{gx}_{gy}" for gx, gy in env.goal_positions]
    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))
            
    clear_set = set(locations)
    for _, loc in boxes: clear_set.discard(loc)
    for _, loc in heavyboxes: clear_set.discard(loc)

    obj_str = "    " + " ".join(locations) + " - location\n"
    if agents: obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes: obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if heavyboxes: obj_str += "    " + " ".join([b[0] for b in heavyboxes]) + " - heavybox\n"
    obj_str += "    right left up down - direction\n"

    init_str = ""
    for loc in clear_set: init_str += f"    (clear {loc})\n"
    for a, loc in agent_locs: init_str += f"    (agent-at {a} {loc})\n"
    for b, loc in boxes: init_str += f"    (box-at {b} {loc})\n"
    for h, loc in heavyboxes: init_str += f"    (heavybox-at {h} {loc})\n"
    for l1, l2, dname in adjacencies:
        init_str += f"    (adj {l1} {l2})\n"
        init_str += f"    (move-dir {l1} {l2} {dname})\n"

    # Brute-force mapping: each box must go to one goal, and each goal must have one box.
    # Since we can't easily do matching in STRIPS, we'll use a simpler heuristic:
    # Any box can go to any goal, but we need all boxes to be at SOME goal.
    # We'll use the scan order for goals but boxes are stable.
    goal_conditions = []
    for i, (b_name, _) in enumerate(boxes):
        if i < len(goals):
            goal_conditions.append(f"(box-at {b_name} {goals[i]})")
    for j, (h_name, _) in enumerate(heavyboxes):
        idx = len(boxes) + j
        if idx < len(goals):
            goal_conditions.append(f"(heavybox-at {h_name} {goals[idx]})")

    goal_str = "(and " + " ".join(goal_conditions) + ")"

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
