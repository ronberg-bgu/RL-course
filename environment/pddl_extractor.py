import os

def generate_domain(domain_path):
    # YOUR exact PDDL domain logic with directional adjacencies
    domain_str = """(define (domain box-push)
  (:requirements :typing :equality)
  (:types agent box heavybox location direction)
  
  (:predicates
    (agent-at ?a - agent ?l - location)
    (box-at ?b - box ?l - location)
    (heavybox-at ?h - heavybox ?l - location)
    (clear ?l - location)
    (adj ?l1 - location ?l2 - location ?d - direction)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location ?d - direction)
    :precondition (and (agent-at ?a ?from) (clear ?to) (adj ?from ?to ?d))
    :effect (and 
      (not (agent-at ?a ?from)) 
      (agent-at ?a ?to)
    )
  )

  (:action push-small
    :parameters (?a - agent ?pos - location ?boxpos - location ?newboxpos - location ?d - direction ?b - box)
    :precondition (and 
      (agent-at ?a ?pos) 
      (box-at ?b ?boxpos) 
      (clear ?newboxpos)
      (adj ?pos ?boxpos ?d) 
      (adj ?boxpos ?newboxpos ?d) 
    )
    :effect (and 
      (not (agent-at ?a ?pos)) 
      (agent-at ?a ?boxpos)
      (not (box-at ?b ?boxpos)) 
      (box-at ?b ?newboxpos)
      (clear ?boxpos)
      (not (clear ?newboxpos))
    )
  )

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?pos - location ?boxpos - location ?newboxpos - location ?d - direction ?h - heavybox)
    :precondition (and 
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?pos) 
      (agent-at ?a2 ?pos) 
      (heavybox-at ?h ?boxpos) 
      (clear ?newboxpos)
      (adj ?pos ?boxpos ?d) 
      (adj ?boxpos ?newboxpos ?d) 
    )
    :effect (and 
      (not (agent-at ?a1 ?pos)) 
      (not (agent-at ?a2 ?pos)) 
      (agent-at ?a1 ?boxpos)
      (agent-at ?a2 ?boxpos)
      (not (heavybox-at ?h ?boxpos)) 
      (heavybox-at ?h ?newboxpos)
      (clear ?boxpos)
      (not (clear ?newboxpos))
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
    
    boxes_coords = []
    heavyboxes_coords = []
    goals_coords = []

    # FIX: Always read goals from the static ascii_map so they don't disappear when covered!
    for y, row in enumerate(env.ascii_map):
        for x, char in enumerate(row):
            if char == 'G':
                goals_coords.append((x, y))

    # Analyze the dynamic grid for locations and boxes
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is None or cell.type != 'wall':
                loc = f"loc_{x}_{y}"
                locations.append(loc)

                # Directional adjacencies
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

                # Extract objects
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        heavyboxes_coords.append((x, y))
                    else:
                        boxes_coords.append((x, y))

    boxes_coords.sort(key=lambda p: (p[0], p[1]))
    goals_coords.sort(key=lambda p: (p[0], p[1]))
    heavyboxes_coords.sort(key=lambda p: (p[0], p[1]))

    # Smart Goal Mapping (Lock boxes already on goals so they don't swap)
    small_goals = goals_coords[:len(boxes_coords)]
    heavy_goals = goals_coords[len(boxes_coords):]
    
    assigned_boxes = {}
    assigned_small_goals = set()
    
    for i, b_pos in enumerate(boxes_coords):
        if b_pos in small_goals:
            assigned_boxes[i] = b_pos
            assigned_small_goals.add(b_pos)
            
    unassigned_goals = [g for g in small_goals if g not in assigned_small_goals]
    g_idx = 0
    for i, b_pos in enumerate(boxes_coords):
        if i not in assigned_boxes:
            assigned_boxes[i] = unassigned_goals[g_idx]
            g_idx += 1

    boxes = [(f"box_{i}", f"loc_{x}_{y}") for i, (x, y) in enumerate(boxes_coords)]
    heavyboxes = [(f"hbx_{i}", f"loc_{x}_{y}") for i, (x, y) in enumerate(heavyboxes_coords)]
    
    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))
            
    clear_set = set(locations)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, loc in heavyboxes:
        clear_set.discard(loc)

    # Inject your custom directional identifiers
    obj_str = "    up down left right - direction\n"
    obj_str += "    " + " ".join(locations) + " - location\n"
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
    for h, loc in heavyboxes:
        init_str += f"    (heavybox-at {h} {loc})\n"
    for l1, l2, d in adjacencies:
        init_str += f"    (adj {l1} {l2} {d})\n"

    goal_conditions = []
    for i in range(len(boxes_coords)):
        gx, gy = assigned_boxes[i]
        goal_conditions.append(f"(box-at box_{i} loc_{gx}_{gy})")
        
    for i in range(len(heavyboxes_coords)):
        # Safely assign heavy goals
        if i < len(heavy_goals):
            gx, gy = heavy_goals[i]
            goal_conditions.append(f"(heavybox-at hbx_{i} loc_{gx}_{gy})")

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