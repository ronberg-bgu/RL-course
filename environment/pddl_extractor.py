import os

# Stable mapping for box objects to names
BOX_REGISTRY = {}

def get_box_name(cell, box_type):
    global BOX_REGISTRY
    mem_address = id(cell)
    if mem_address in BOX_REGISTRY:
        return BOX_REGISTRY[mem_address]
    prefix = "hbx" if box_type == "heavy" else "box"
    assigned_name = f"{prefix}_{len(BOX_REGISTRY)}"
    BOX_REGISTRY[mem_address] = assigned_name
    return assigned_name

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
    :precondition (and
        (agent-at ?a ?from)
        (adj ?from ?to)
        (clear ?to)
    )
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?d - direction)
    :precondition (and
        (agent-at ?a ?from)
        (move-dir ?from ?boxloc ?d)
        (box-at ?b ?boxloc)
        (move-dir ?boxloc ?toloc ?d)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a ?boxloc)
        (not (agent-at ?a ?from))
        (box-at ?b ?toloc)
        (not (box-at ?b ?boxloc))
        (not (clear ?toloc))
        (clear ?boxloc)
    )
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


def _build_objects(locations, agents, boxes, heavyboxes):
    parts = [" ".join(locations) + " - location"]
    if agents:
        parts.append(" ".join(agents) + " - agent")
    if boxes:
        parts.append(" ".join(b[0] for b in boxes) + " - box")
    if heavyboxes:
        parts.append(" ".join(h[0] for h in heavyboxes) + " - heavybox")
    parts.append("right left up down - direction")
    return "".join(f"    {p}\n" for p in parts)


def _build_init(clear_set, agent_locs, boxes, heavyboxes, adjacencies):
    lines = []
    for loc in clear_set:
        lines.append(f"(clear {loc})")
    for a, loc in agent_locs:
        lines.append(f"(agent-at {a} {loc})")
    for b, loc in boxes:
        lines.append(f"(box-at {b} {loc})")
    for h, loc in heavyboxes:
        lines.append(f"(heavybox-at {h} {loc})")
    for l1, l2, dname in adjacencies:
        lines.append(f"(adj {l1} {l2})")
        lines.append(f"(move-dir {l1} {l2} {dname})")
    return "".join(f"    {l}\n" for l in lines)


def _build_goal(boxes, heavyboxes, goals):
    conditions = []
    for i, (b_name, _) in enumerate(boxes):
        if i < len(goals):
            conditions.append(f"(box-at {b_name} {goals[i]})")
    for j, (h_name, _) in enumerate(heavyboxes):
        idx = len(boxes) + j
        if idx < len(goals):
            conditions.append(f"(heavybox-at {h_name} {goals[idx]})")
    return "(and " + " ".join(conditions) + ")"


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
                    if b_type == "heavy":
                        heavyboxes.append((b_name, loc))
                    else:
                        boxes.append((b_name, loc))

    goals = [f"loc_{gx}_{gy}" for gx, gy in env.goal_positions]
    agent_locs = [(a, f"loc_{env.agent_positions[a][0]}_{env.agent_positions[a][1]}") for a in agents]

    clear_set = set(locations)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, loc in heavyboxes:
        clear_set.discard(loc)

    obj_str  = _build_objects(locations, agents, boxes, heavyboxes)
    init_str = _build_init(clear_set, agent_locs, boxes, heavyboxes, adjacencies)
    goal_str = _build_goal(boxes, heavyboxes, goals)

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