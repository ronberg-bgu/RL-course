import os

def generate_domain(domain_path):
    domain_str = """(define (domain box-push)
  (:requirements :strips :typing)
  (:types agent location box)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (goal ?loc - location)
  )
  
  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)) (not (clear ?to)) (clear ?from))
  )
  
  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and (agent-at ?a ?from) (adj ?from ?boxloc) (box-at ?b ?boxloc) (adj ?boxloc ?toloc) (clear ?toloc))
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)))
  )
)
"""
    with open(domain_path, "w") as f:
        f.write(domain_str)

def generate_problem(env, problem_path):
    # Extract locations, agents, boxes from env grid
    w, h = env.width, env.height
    
    locations = []
    adjacencies = []
    clear_locs = []
    agents = env.agents
    boxes = []
    goals = []
    
    # Analyze the static grid
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is None or cell.type != 'wall':
                loc = f"loc_{x}_{y}"
                locations.append(loc)
                
                # Check adjacencies (right and down to avoid repeating, then double it)
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
                        
                # Determine what is here
                if cell is not None and cell.type == "goal":
                    goals.append(loc)
                elif cell is not None and cell.type == "box":
                    boxes.append((f"box_{len(boxes)}", loc))
                else:
                    # Initially assume clear, we will remove agent locations later
                    clear_locs.append(loc)

    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        aloc = f"loc_{px}_{py}"
        agent_locs.append((a, aloc))
        if aloc in clear_locs:
            clear_locs.remove(aloc)
            
    # For boxes, they were actually in the grid so they might be in clear_locs?
    # Wait, my logic above added loc to clear_locs if not goal and not box.
    # What if it IS goal? Goal is clear but has a Goal object!
    # Let's rebuild clear_locs reliably
    clear_set = set(locations)
    for _, loc in agent_locs:
        clear_set.discard(loc)
    for _, loc in boxes:
        clear_set.discard(loc)

    obj_str = "    " + " ".join(locations) + " - location\n"
    if agents:
        obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes:
        obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"

    init_str = ""
    for loc in clear_set:
        init_str += f"    (clear {loc})\n"
    for a, loc in agent_locs:
        init_str += f"    (agent-at {a} {loc})\n"
    for b, loc in boxes:
        init_str += f"    (box-at {b} {loc})\n"
    for l1, l2 in adjacencies:
        init_str += f"    (adj {l1} {l2})\n"
        
    goal_str = ""
    if goals and boxes:
        # Simplistic goal: any box at the goal
        goal_str = f"(box-at {boxes[0][0]} {goals[0]})"
    else:
        goal_str = "(clear loc_1_1) ; dummy"

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
