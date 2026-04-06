
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

# Now these imports will work without crashing!
from visualize_plan import visualize_pddl_plan
from exercises.ex1.pddl_to_map import parse_pddl_to_map 


# ==============================================================================
# LLM PROMPTING STRATEGY 
# ==============================================================================
PROMPT_STRATEGY = """
Act as an expert in classical planning and PDDL. Your task is to write a strictly compliant `domain.pddl` and `problem.pddl` for a Box Pushing puzzle.

**The World Rules**
* There are two agents moving on a 2D grid.
* Agents can move in any of the 4 directions to adjacent empty cells (agents may share a cell).
* Agents can push a regular box (requires 1 agent) or a heavy box (requires 2 agents).
* To push the heavy box, both agents must push simultaneously from the same cell in the same direction.
* The goal condition must refer only to the final positions of the boxes, not the agents.

**Domain Constraints (STRICT)**
You must use the following types and predicates exactly as written. Do not invent new ones.
* Types: `agent`, `location`, `box`, `heavybox`.
* Predicates:
  * `(agent-at ?a - agent ?loc - location)`
  * `(box-at ?b - box ?loc - location)`
  * `(heavybox-at ?h - heavybox ?loc - location)`
  * `(clear ?loc - location)` - meaning the location has no box on it.
  * `(adj ?l1 - location ?l2 - location)` - meaning locations are adjacent (bidirectional).

**Action Definitions**
Define exactly 3 actions:
1. `move`: Parameters `(?a - agent ?from - location ?to - location)`. Moves one agent to an adjacent cell.
2. `push-small`: Parameters `(?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)`. One agent pushes a small box one cell.
3. `push-heavy`: You MUST use this exact PDDL for the heavy push:
(:action push-heavy
  :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
  :precondition (and
    (not (= ?a1 ?a2))
    (agent-at ?a1 ?from) (agent-at ?a2 ?from)
    (adj ?from ?boxloc) (heavybox-at ?h ?boxloc)
    (adj ?boxloc ?toloc) (clear ?toloc))
  :effect (and
    (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
    (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
    (clear ?from)
    (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc))
    (not (clear ?toloc)))
)

**Problem Requirements**
Create a problem file for a 4x4 grid (X=1..4, Y=1..4).
* Object Naming Convention:
  * Locations: `loc_X_Y` (e.g., `loc_1_1` to `loc_4_4`).
  * Agents: `agent_0`, `agent_1`.
  * Small boxes: `box_0`, `box_1`.
  * Heavy box: `hbx_0`.
* Initial State (`:init`):
  * Define the bidirectional adjacency (`adj`) for the 4x4 grid.
  * Set all locations as `(clear)` EXCEPT where boxes are placed initially.
  * Agents: `agent_0` at `loc_1_1`, `agent_1` at `loc_2_1`.
  * Small boxes: `box_0` at `loc_1_3`, `box_1` at `loc_4_3` (do NOT mark these as clear).
  * Heavy box: `hbx_0` at `loc_2_3` (do NOT mark this as clear).
* Goal State (`:goal`):
  * `box_0` at `loc_1_4`.
  * `box_1` at `loc_4_4`.
  * `hbx_0` at `loc_2_4`.

Output the `domain.pddl` and `problem.pddl` in two separate Lisp/PDDL code blocks. Do not add extra explanations.
"""
# ==============================================================================

DOMAIN_PDDL = """
(define (domain box-pushing)
  (:requirements :strips :typing :negative-preconditions :equality)
  
  (:types
    agent location box heavybox
  )

  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and
      (agent-at ?a ?from)
      (adj ?from ?to)
      (clear ?to))
    :effect (and
      (agent-at ?a ?to)
      (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (adj ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj ?boxloc ?toloc)
      (clear ?toloc))
    :effect (and
      (agent-at ?a ?boxloc)
      (not (agent-at ?a ?from))
      (box-at ?b ?toloc)
      (not (box-at ?b ?boxloc))
      (clear ?boxloc)
      (not (clear ?toloc)))
  )

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from) (agent-at ?a2 ?from)
      (adj ?from ?boxloc) (heavybox-at ?h ?boxloc)
      (adj ?boxloc ?toloc) (clear ?toloc))
    :effect (and
      (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
      (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
      (clear ?from)
      (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc))
      (not (clear ?toloc)))
  )
)
"""

PROBLEM_PDDL = """
(define (problem box-pushing-4x4)
  (:domain box-pushing)

  (:objects
    agent_0 agent_1 - agent
    loc_1_1 loc_1_2 loc_1_3 loc_1_4
    loc_2_1 loc_2_2 loc_2_3 loc_2_4
    loc_3_1 loc_3_2 loc_3_3 loc_3_4
    loc_4_1 loc_4_2 loc_4_3 loc_4_4 - location
    box_0 box_1 - box
    hbx_0 - heavybox
  )

  (:init
    ;; Adjacency
    (adj loc_1_1 loc_1_2) (adj loc_1_2 loc_1_1)
    (adj loc_1_2 loc_1_3) (adj loc_1_3 loc_1_2)
    (adj loc_1_3 loc_1_4) (adj loc_1_4 loc_1_3)

    (adj loc_2_1 loc_2_2) (adj loc_2_2 loc_2_1)
    (adj loc_2_2 loc_2_3) (adj loc_2_3 loc_2_2)
    (adj loc_2_3 loc_2_4) (adj loc_2_4 loc_2_3)

    (adj loc_3_1 loc_3_2) (adj loc_3_2 loc_3_1)
    (adj loc_3_2 loc_3_3) (adj loc_3_3 loc_3_2)
    (adj loc_3_3 loc_3_4) (adj loc_3_4 loc_3_3)

    (adj loc_4_1 loc_4_2) (adj loc_4_2 loc_4_1)
    (adj loc_4_2 loc_4_3) (adj loc_4_3 loc_4_2)
    (adj loc_4_3 loc_4_4) (adj loc_4_4 loc_4_3)

    (adj loc_1_1 loc_2_1) (adj loc_2_1 loc_1_1)
    (adj loc_2_1 loc_3_1) (adj loc_3_1 loc_2_1)
    (adj loc_3_1 loc_4_1) (adj loc_4_1 loc_3_1)

    (adj loc_1_2 loc_2_2) (adj loc_2_2 loc_1_2)
    (adj loc_2_2 loc_3_2) (adj loc_3_2 loc_2_2)
    (adj loc_3_2 loc_4_2) (adj loc_4_2 loc_3_2)

    (adj loc_1_3 loc_2_3) (adj loc_2_3 loc_1_3)
    (adj loc_2_3 loc_3_3) (adj loc_3_3 loc_2_3)
    (adj loc_3_3 loc_4_3) (adj loc_4_3 loc_3_3)

    (adj loc_1_4 loc_2_4) (adj loc_2_4 loc_1_4)
    (adj loc_2_4 loc_3_4) (adj loc_3_4 loc_2_4)
    (adj loc_3_4 loc_4_4) (adj loc_4_4 loc_3_4)

    ;; Clear locations (all except box locations)
    (clear loc_1_1)
    (clear loc_1_2)
    (clear loc_1_4)
    (clear loc_2_1)
    (clear loc_2_2)
    (clear loc_2_4)
    (clear loc_3_1)
    (clear loc_3_2)
    (clear loc_3_3)
    (clear loc_3_4)
    (clear loc_4_1)
    (clear loc_4_2)
    (clear loc_4_4)

    ;; Agent positions
    (agent-at agent_0 loc_1_1)
    (agent-at agent_1 loc_2_1)

    ;; Box positions
    (box-at box_0 loc_1_3)
    (box-at box_1 loc_4_3)
    (heavybox-at hbx_0 loc_2_3)
  )

  (:goal
    (and
      (box-at box_0 loc_1_4)
      (box-at box_1 loc_4_4)
      (heavybox-at hbx_0 loc_2_4))
  )
)
"""

def main():
    # 1. Ensure the pddl output directory exists inside the current folder (ex1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pddl_dir = os.path.join(current_dir, "pddl")
    os.makedirs(pddl_dir, exist_ok=True)

    domain_path = os.path.join(pddl_dir, "domain.pddl")
    problem_path = os.path.join(pddl_dir, "problem.pddl")

    # 2. Write embedded strings to PDDL files
    with open(domain_path, "w") as f:
        f.write(DOMAIN_PDDL.strip())
    
    with open(problem_path, "w") as f:
        f.write(PROBLEM_PDDL.strip())

    print(f"[*] Successfully generated {domain_path} and {problem_path}")

    # 3. Parse PDDL back into the ASCII map required by the visualizer
    print("[*] Translating PDDL to ASCII Map...")
    ascii_map = parse_pddl_to_map(domain_path, problem_path)
    
    print("\nReconstructed ASCII Map:")
    for row in ascii_map:
        print(row)
    print("\n")

    # 4. Invoke the visualizer
    print("[*] Starting Planner and Visualizer...")
    visualize_pddl_plan(ascii_map, domain_path, problem_path)

if __name__ == "__main__":
    main()




