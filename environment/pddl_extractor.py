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
    (inline ?l1 - location ?l2 - location ?l3 - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and (agent-at ?a ?from) (adj ?from ?boxloc) (box-at ?b ?boxloc) (adj ?boxloc ?toloc) (clear ?toloc) (inline ?from ?boxloc ?toloc))
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)))
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
        (inline ?from ?boxloc ?toloc)
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

def generate_problem(env, problem_path):
    # Extract locations, agents, boxes from env grid
    w, h = env.width, env.height

    # Goal positions that already have a box on them (box overwrites goal in grid)
    original_goal_positions = getattr(env, 'goal_positions', frozenset())

    locations = []
    adjacencies = []
    inlines = []
    agents = env.agents
    # Each entry: (name, loc, is_done) — is_done=True means box already at a goal
    boxes = []
    heavyboxes = []
    free_goals = []   # goal locations that still have no box on them

    # Analyze the static grid
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is None or cell.type != 'wall':
                loc = f"loc_{x}_{y}"
                locations.append(loc)

                # Check adjacencies (undirected)
                if x < w - 1:
                    r_cell = env.core_env.grid.get(x+1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x+1}_{y}"))
                        adjacencies.append((f"loc_{x+1}_{y}", loc))
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y+1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x}_{y+1}"))
                        adjacencies.append((f"loc_{x}_{y+1}", loc))

                # Inline triples: (x,y) as the middle cell, same row or same column.
                # Horizontal: (x-1,y) — (x,y) — (x+1,y)
                if x > 0 and x < w - 1:
                    l_cell = env.core_env.grid.get(x-1, y)
                    r_cell2 = env.core_env.grid.get(x+1, y)
                    if (l_cell is None or l_cell.type != 'wall') and \
                       (r_cell2 is None or r_cell2.type != 'wall'):
                        l1, l3 = f"loc_{x-1}_{y}", f"loc_{x+1}_{y}"
                        inlines.append((l1, loc, l3))
                        inlines.append((l3, loc, l1))
                # Vertical: (x,y-1) — (x,y) — (x,y+1)
                if y > 0 and y < h - 1:
                    u_cell = env.core_env.grid.get(x, y-1)
                    d_cell2 = env.core_env.grid.get(x, y+1)
                    if (u_cell is None or u_cell.type != 'wall') and \
                       (d_cell2 is None or d_cell2.type != 'wall'):
                        l1, l3 = f"loc_{x}_{y-1}", f"loc_{x}_{y+1}"
                        inlines.append((l1, loc, l3))
                        inlines.append((l3, loc, l1))

                # Determine what is here
                if cell is not None and cell.type == "goal":
                    free_goals.append(loc)
                elif cell is not None and cell.type == "box":
                    is_done = (x, y) in original_goal_positions
                    if getattr(cell, "box_size", "") == "heavy":
                        heavyboxes.append((f"hbx_{len(heavyboxes)}", loc, is_done))
                    else:
                        boxes.append((f"box_{len(boxes)}", loc, is_done))

    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))

    # clear = no box/heavybox occupying this cell; agents don't affect clear
    # so two agents can share a cell (required for push-heavy precondition)
    clear_set = set(locations)
    for _, loc, _ in boxes:
        clear_set.discard(loc)
    for _, loc, _ in heavyboxes:
        clear_set.discard(loc)

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
    for b_name, loc, _ in boxes:
        init_str += f"    (box-at {b_name} {loc})\n"
    for h_name, loc, _ in heavyboxes:
        init_str += f"    (heavybox-at {h_name} {loc})\n"
    for l1, l2 in adjacencies:
        init_str += f"    (adj {l1} {l2})\n"
    for l1, l2, l3 in inlines:
        init_str += f"    (inline {l1} {l2} {l3})\n"

    # Build goal conditions:
    # - done boxes/heavyboxes keep their current location as goal (already satisfied)
    # - todo boxes/heavyboxes are paired with remaining free goal locations in scan order
    goal_conditions = []
    free_goal_idx = 0
    for b_name, loc, is_done in boxes:
        if is_done:
            goal_conditions.append(f"(box-at {b_name} {loc})")
        elif free_goal_idx < len(free_goals):
            goal_conditions.append(f"(box-at {b_name} {free_goals[free_goal_idx]})")
            free_goal_idx += 1
    for h_name, loc, is_done in heavyboxes:
        if is_done:
            goal_conditions.append(f"(heavybox-at {h_name} {loc})")
        elif free_goal_idx < len(free_goals):
            goal_conditions.append(f"(heavybox-at {h_name} {free_goals[free_goal_idx]})")
            free_goal_idx += 1

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
