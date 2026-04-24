import os


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
    (aligned ?l1 - location ?l2 - location ?l3 - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and 
        (agent-at ?a ?from) 
        (adj ?from ?boxloc) 
        (box-at ?b ?boxloc) 
        (adj ?boxloc ?toloc) 
        (clear ?toloc) 
        (aligned ?from ?boxloc ?toloc)
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
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from)
        (agent-at ?a2 ?from)
        (adj ?from ?boxloc)
        (heavybox-at ?h ?boxloc)
        (adj ?boxloc ?toloc)
        (clear ?toloc)
        (aligned ?from ?boxloc ?toloc)
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
                    r_cell = env.core_env.grid.get(x + 1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x + 1}_{y}"))
                        adjacencies.append((f"loc_{x + 1}_{y}", loc))
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y + 1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x}_{y + 1}"))
                        adjacencies.append((f"loc_{x}_{y + 1}", loc))

                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        heavyboxes.append((f"hbx_{len(heavyboxes)}", loc))
                    else:
                        boxes.append((f"box_{len(boxes)}", loc))

    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))

    clear_set = set(locations)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, loc in heavyboxes:
        clear_set.discard(loc)

    aligned_triplets = []
    for y in range(h):
        for x in range(w - 2):
            l1, l2, l3 = f"loc_{x}_{y}", f"loc_{x + 1}_{y}", f"loc_{x + 2}_{y}"
            if l1 in locations and l2 in locations and l3 in locations:
                aligned_triplets.append((l1, l2, l3))
                aligned_triplets.append((l3, l2, l1))
    for x in range(w):
        for y in range(h - 2):
            l1, l2, l3 = f"loc_{x}_{y}", f"loc_{x}_{y + 1}", f"loc_{x}_{y + 2}"
            if l1 in locations and l2 in locations and l3 in locations:
                aligned_triplets.append((l1, l2, l3))
                aligned_triplets.append((l3, l2, l1))

    obj_str = "    " + " ".join(locations) + " - location\n"
    if agents:
        obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes:
        obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if heavyboxes:
        obj_str += "    " + " ".join([b[0] for b in heavyboxes]) + " - heavybox\n"

    init_str = ""
    for loc in clear_set:
        init_str += f"    (clear {loc})\n"
    for a, loc in agent_locs:
        init_str += f"    (agent-at {a} {loc})\n"
    for b, loc in boxes:
        init_str += f"    (box-at {b} {loc})\n"
    for h_box, loc in heavyboxes:
        init_str += f"    (heavybox-at {h_box} {loc})\n"
    for l1, l2 in adjacencies:
        init_str += f"    (adj {l1} {l2})\n"
    for l1, l2, l3 in aligned_triplets:
        init_str += f"    (aligned {l1} {l2} {l3})\n"

    # ── FIX: ANTI-SWAP GOAL ASSIGNMENT ──
    goal_conditions = []
    goal_y = h - 2

    for h_name, _ in heavyboxes:
        goal_conditions.append(f"(heavybox-at {h_name} loc_2_{goal_y})")

    # Strictly sort the boxes by X coordinate (index 1 of the split string)
    def get_x(b_tuple):
        return int(b_tuple[1].split('_')[1])

    sorted_small_boxes = sorted(boxes, key=get_x)

    small_xs = [1, 3]
    for i, (b_name, _) in enumerate(sorted_small_boxes):
        if i < len(small_xs):
            goal_conditions.append(f"(box-at {b_name} loc_{small_xs[i]}_{goal_y})")

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