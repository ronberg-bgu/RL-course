"""
llm_pipeline.py — HW1: LLM-Generated PDDL Pipeline
====================================================

This script generates PDDL domain and problem files for a multi-agent
box-pushing problem using content produced by an LLM (Claude).

Approach: The PDDL was generated once using Claude by providing it with
a detailed prompt describing the world, the PDDL specification, and our
custom map layout. The LLM-generated content is embedded below and written
to pddl/domain.pddl and pddl/problem.pddl on each run.

The script then:
  1. Writes the LLM-generated PDDL files to disk.
  2. Calls pddl_to_map.py to reconstruct the ASCII map from the PDDL.
  3. Calls the visualizer to solve and display the plan.

Usage:
    python exercises/ex1/llm_pipeline.py 2>&1 | tee exercises/ex1/planner_output.txt
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PDDL_FOLDER = os.path.join(SCRIPT_DIR, "pddl")

# ===========================================================================
# LLM PROMPTING STRATEGY
# ===========================================================================
# The following prompt was sent to Claude (Anthropic) to generate the PDDL
# domain and problem files. The prompt strategy was:
#
#   1. Provide the EXACT PDDL specification from the assignment (types,
#      predicates, action signatures, naming conventions) so the LLM
#      produces simulator-compatible output.
#
#   2. Describe our custom map in both ASCII art and natural language
#      so the LLM can accurately enumerate locations, adjacencies,
#      initial positions, and goal conditions.
#
#   3. Ask the LLM to output ONLY the raw PDDL content for each file,
#      with no surrounding explanation.
#
# ------------- PROMPT SENT TO LLM (Claude) ---------------------------------
#
# """
# I need you to generate two PDDL files (domain.pddl and problem.pddl) for a
# multi-agent box-pushing planning problem. Here is the EXACT specification
# you must follow:
#
# ## Types
#   agent   location   box   heavybox
#
# ## Predicates
#   (agent-at ?a - agent ?loc - location)
#   (box-at ?b - box ?loc - location)
#   (heavybox-at ?h - heavybox ?loc - location)
#   (clear ?loc - location)
#   (adj ?l1 - location ?l2 - location)
#
# ## Actions
#   move(?a, ?from, ?to) — one agent moves to adjacent cell
#   push-small(?a, ?from, ?boxloc, ?toloc, ?b) — one agent pushes small box
#   push-heavy(?a1, ?a2, ?from, ?boxloc, ?toloc, ?h) — two agents push heavy
#     box from SAME cell, SAME direction simultaneously
#
# ## push-heavy details
#   Both agents must be at the same cell (?from), adjacent to the heavybox.
#   Both agents move into ?boxloc, the heavybox moves to ?toloc.
#   Precondition includes (not (= ?a1 ?a2)).
#
# ## Naming conventions
#   Locations: loc_X_Y  (where X=column, Y=row, 0-indexed)
#   Agents: agent_0, agent_1
#   Small boxes: box_0, box_1
#   Heavy box: hbx_0
#
# ## My custom map (8x8):
#
#   Row 0: WWWWWWWW
#   Row 1: W  AA  W    <- agents at (3,1) and (4,1)
#   Row 2: W  C   W    <- heavy box at (3,2)
#   Row 3: W W  B W    <- wall at (2,3), small box at (5,3)
#   Row 4: W W  B W    <- wall at (2,4), small box at (5,4)
#   Row 5: W WGGG W    <- wall at (2,5), goals at (3,5), (4,5), (5,5)
#   Row 6: W      W
#   Row 7: WWWWWWWW
#
# The outer boundary (row 0, row 7, col 0, col 7) is all walls.
# Internal walls at (2,3), (2,4), and (2,5) form a corridor barrier.
#
# Non-wall locations are every (x,y) that is NOT a wall. Agents and boxes
# occupy locations but those locations still exist.
#
# Adjacency: two locations are adjacent if they differ by exactly 1 in
# either x or y (not both) AND neither is a wall. Adjacency is bidirectional.
#
# Clear: a location is clear if it has NO box and NO heavybox on it.
# Agents do NOT block locations from being clear.
#
# Goal: all 3 boxes must reach the goal cells:
#   box_0 -> loc_3_5
#   box_1 -> loc_4_5
#   hbx_0 -> loc_5_5
#
# Please output the domain.pddl content first, then the problem.pddl content.
# Output ONLY raw PDDL, no markdown fences or explanations.
# """
#
# ------------- END OF PROMPT ------------------------------------------------

# ===========================================================================
# LLM-GENERATED PDDL CONTENT (produced by Claude from the prompt above)
# ===========================================================================

DOMAIN_PDDL = """\
(define (domain box-push)
  (:requirements :strips :typing :equality)
  (:types agent location box heavybox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and (agent-at ?a ?from) (adj ?from ?boxloc) (box-at ?b ?boxloc) (adj ?boxloc ?toloc) (clear ?toloc))
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

PROBLEM_PDDL = """\
(define (problem bp-map)
  (:domain box-push)
  (:objects
    loc_1_1 loc_2_1 loc_3_1 loc_4_1 loc_5_1 loc_6_1 loc_1_2 loc_2_2 loc_3_2 loc_4_2 loc_5_2 loc_6_2 loc_1_3 loc_3_3 loc_4_3 loc_5_3 loc_6_3 loc_1_4 loc_3_4 loc_4_4 loc_5_4 loc_6_4 loc_1_5 loc_3_5 loc_4_5 loc_5_5 loc_6_5 loc_1_6 loc_2_6 loc_3_6 loc_4_6 loc_5_6 loc_6_6 - location
    agent_0 agent_1 - agent
    box_0 box_1 - box
    hbx_0 - heavybox
  )
  (:init
    (clear loc_6_2)
    (clear loc_1_3)
    (clear loc_1_6)
    (clear loc_5_2)
    (clear loc_6_3)
    (clear loc_5_6)
    (clear loc_5_5)
    (clear loc_1_5)
    (clear loc_3_4)
    (clear loc_1_1)
    (clear loc_2_1)
    (clear loc_1_2)
    (clear loc_3_5)
    (clear loc_4_5)
    (clear loc_6_5)
    (clear loc_1_4)
    (clear loc_2_2)
    (clear loc_4_3)
    (clear loc_6_6)
    (clear loc_6_1)
    (clear loc_4_2)
    (clear loc_4_6)
    (clear loc_5_1)
    (clear loc_6_4)
    (clear loc_4_4)
    (clear loc_3_3)
    (clear loc_2_6)
    (clear loc_3_6)
    (agent-at agent_0 loc_3_1)
    (agent-at agent_1 loc_4_1)
    (box-at box_0 loc_5_3)
    (box-at box_1 loc_5_4)
    (heavybox-at hbx_0 loc_3_2)
    (adj loc_1_1 loc_2_1)
    (adj loc_2_1 loc_1_1)
    (adj loc_1_1 loc_1_2)
    (adj loc_1_2 loc_1_1)
    (adj loc_2_1 loc_3_1)
    (adj loc_3_1 loc_2_1)
    (adj loc_2_1 loc_2_2)
    (adj loc_2_2 loc_2_1)
    (adj loc_3_1 loc_4_1)
    (adj loc_4_1 loc_3_1)
    (adj loc_3_1 loc_3_2)
    (adj loc_3_2 loc_3_1)
    (adj loc_4_1 loc_5_1)
    (adj loc_5_1 loc_4_1)
    (adj loc_4_1 loc_4_2)
    (adj loc_4_2 loc_4_1)
    (adj loc_5_1 loc_6_1)
    (adj loc_6_1 loc_5_1)
    (adj loc_5_1 loc_5_2)
    (adj loc_5_2 loc_5_1)
    (adj loc_6_1 loc_6_2)
    (adj loc_6_2 loc_6_1)
    (adj loc_1_2 loc_2_2)
    (adj loc_2_2 loc_1_2)
    (adj loc_1_2 loc_1_3)
    (adj loc_1_3 loc_1_2)
    (adj loc_2_2 loc_3_2)
    (adj loc_3_2 loc_2_2)
    (adj loc_3_2 loc_4_2)
    (adj loc_4_2 loc_3_2)
    (adj loc_3_2 loc_3_3)
    (adj loc_3_3 loc_3_2)
    (adj loc_4_2 loc_5_2)
    (adj loc_5_2 loc_4_2)
    (adj loc_4_2 loc_4_3)
    (adj loc_4_3 loc_4_2)
    (adj loc_5_2 loc_6_2)
    (adj loc_6_2 loc_5_2)
    (adj loc_5_2 loc_5_3)
    (adj loc_5_3 loc_5_2)
    (adj loc_6_2 loc_6_3)
    (adj loc_6_3 loc_6_2)
    (adj loc_1_3 loc_1_4)
    (adj loc_1_4 loc_1_3)
    (adj loc_3_3 loc_4_3)
    (adj loc_4_3 loc_3_3)
    (adj loc_3_3 loc_3_4)
    (adj loc_3_4 loc_3_3)
    (adj loc_4_3 loc_5_3)
    (adj loc_5_3 loc_4_3)
    (adj loc_4_3 loc_4_4)
    (adj loc_4_4 loc_4_3)
    (adj loc_5_3 loc_6_3)
    (adj loc_6_3 loc_5_3)
    (adj loc_5_3 loc_5_4)
    (adj loc_5_4 loc_5_3)
    (adj loc_6_3 loc_6_4)
    (adj loc_6_4 loc_6_3)
    (adj loc_1_4 loc_1_5)
    (adj loc_1_5 loc_1_4)
    (adj loc_3_4 loc_4_4)
    (adj loc_4_4 loc_3_4)
    (adj loc_3_4 loc_3_5)
    (adj loc_3_5 loc_3_4)
    (adj loc_4_4 loc_5_4)
    (adj loc_5_4 loc_4_4)
    (adj loc_4_4 loc_4_5)
    (adj loc_4_5 loc_4_4)
    (adj loc_5_4 loc_6_4)
    (adj loc_6_4 loc_5_4)
    (adj loc_5_4 loc_5_5)
    (adj loc_5_5 loc_5_4)
    (adj loc_6_4 loc_6_5)
    (adj loc_6_5 loc_6_4)
    (adj loc_1_5 loc_1_6)
    (adj loc_1_6 loc_1_5)
    (adj loc_3_5 loc_4_5)
    (adj loc_4_5 loc_3_5)
    (adj loc_3_5 loc_3_6)
    (adj loc_3_6 loc_3_5)
    (adj loc_4_5 loc_5_5)
    (adj loc_5_5 loc_4_5)
    (adj loc_4_5 loc_4_6)
    (adj loc_4_6 loc_4_5)
    (adj loc_5_5 loc_6_5)
    (adj loc_6_5 loc_5_5)
    (adj loc_5_5 loc_5_6)
    (adj loc_5_6 loc_5_5)
    (adj loc_6_5 loc_6_6)
    (adj loc_6_6 loc_6_5)
    (adj loc_1_6 loc_2_6)
    (adj loc_2_6 loc_1_6)
    (adj loc_2_6 loc_3_6)
    (adj loc_3_6 loc_2_6)
    (adj loc_3_6 loc_4_6)
    (adj loc_4_6 loc_3_6)
    (adj loc_4_6 loc_5_6)
    (adj loc_5_6 loc_4_6)
    (adj loc_5_6 loc_6_6)
    (adj loc_6_6 loc_5_6)
  )
  (:goal
    (and
    (box-at box_0 loc_3_5)
    (box-at box_1 loc_4_5)
    (heavybox-at hbx_0 loc_5_5)
  )
  )
)
"""


def write_pddl_files():
    """Write the LLM-generated PDDL content to disk."""
    os.makedirs(PDDL_FOLDER, exist_ok=True)

    domain_path = os.path.join(PDDL_FOLDER, "domain.pddl")
    problem_path = os.path.join(PDDL_FOLDER, "problem.pddl")

    with open(domain_path, "w") as f:
        f.write(DOMAIN_PDDL)
    with open(problem_path, "w") as f:
        f.write(PROBLEM_PDDL)

    return domain_path, problem_path


def main():
    print("=" * 60)
    print("  HW1 — LLM-Generated PDDL Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Write LLM-generated PDDL files to disk
    # ------------------------------------------------------------------
    print("\n[1] Writing LLM-generated PDDL files ...")
    domain_path, problem_path = write_pddl_files()
    print(f"    domain  → {domain_path}")
    print(f"    problem → {problem_path}")

    # ------------------------------------------------------------------
    # Step 2: Reconstruct ASCII map from PDDL (pddl_to_map.py)
    # ------------------------------------------------------------------
    print("\n[2] Reconstructing ASCII map from PDDL files ...")
    from exercises.ex1.pddl_to_map import parse_pddl_to_map

    ascii_map = parse_pddl_to_map(domain_path, problem_path)
    print("    Reconstructed map:")
    for i, row in enumerate(ascii_map):
        print(f"      Row {i}: {row}")

    # ------------------------------------------------------------------
    # Step 3: Solve and visualize
    # ------------------------------------------------------------------
    print("\n[3] Solving PDDL and launching visual simulation ...")
    from visualize_plan import visualize_pddl_plan

    visualize_pddl_plan(ascii_map, domain_path, problem_path)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
