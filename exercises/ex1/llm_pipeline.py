"""
llm_pipeline.py — LLM-based PDDL generation pipeline for Exercise 1.

Authors: Eliad Bazak & Ben Epstein
LLM used: Claude
"""

# Imports
import os
import sys


# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Imports from project
from exercises.ex1.pddl_to_map import pddl_to_ascii_map
from planner.pddl_solver import solve_pddl
from visualize_plan import visualize_pddl_plan

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXERCISE_DIR = os.path.dirname(os.path.abspath(__file__))
PDDL_DIR = os.path.join(EXERCISE_DIR, "pddl")
DOMAIN_FILE = os.path.join(PDDL_DIR, "domain.pddl")
PROBLEM_FILE = os.path.join(PDDL_DIR, "problem.pddl")


# ---------------------------------------------------------------------------
# LLM Prompt (used to generate the PDDL below — kept for documentation) (Claude Opus 4.6)
# ---------------------------------------------------------------------------
LLM_PROMPT = """
Generate PDDL domain and problem files for a multi-agent
box-pushing grid world.

## Goal

Two agents must cooperate to push boxes onto goal cells. One agent can push a
small box alone; a heavy box requires both agents at the same cell to push.
Generate a PDDL domain (action rules) and problem (specific scenario) that a
classical STRIPS planner can solve.

## World Rules

- Agents move to any adjacent (up/down/left/right) non-wall cell that has no
  box or heavybox. Agents do NOT block each other or affect cell clearance —
  only boxes and heavyboxes make a cell not-clear.
- One agent pushes a small box by walking into it, but ONLY in a straight line
  (agent, box, and destination must be collinear). Since STRIPS lacks
  disjunction, use an `in-line` predicate to enforce this:
  ```
  (:action push-small
    :parameters (?a - agent ?from - location ?bloc - location ?to - location ?b - box)
    :precondition (and (agent-at ?a ?from) (box-at ?b ?bloc)
        (adj ?from ?bloc) (adj ?bloc ?to) (in-line ?from ?bloc ?to) (clear ?to))
    :effect (and (agent-at ?a ?bloc) (not (agent-at ?a ?from)) (clear ?from)
        (box-at ?b ?to) (not (box-at ?b ?bloc)) (clear ?bloc) (not (clear ?to))))
  ```
- A heavy box requires BOTH agents at the SAME cell. Use four directional
  actions (push-heavy-up/down/left/right) with directional adjacency predicates
  to enforce consistent push direction. Example:
  ```
  (:action push-heavy-down
    :parameters (?a1 - agent ?a2 - agent ?from - location ?bloc - location ?to - location ?h - heavybox)
    :precondition (and (not (= ?a1 ?a2))
        (agent-at ?a1 ?from) (agent-at ?a2 ?from)
        (adj-down ?from ?bloc) (heavybox-at ?h ?bloc)
        (adj-down ?bloc ?to) (clear ?to))
    :effect (and (agent-at ?a1 ?bloc) (agent-at ?a2 ?bloc)
        (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from)) (clear ?from)
        (heavybox-at ?h ?to) (not (heavybox-at ?h ?bloc)) (clear ?bloc) (not (clear ?to))))
  ```
  Create all four directions, each using its matching adj-{direction} for both
  from→bloc and bloc→to.
- When a box/heavybox is pushed, the vacated cell becomes clear and the
  destination becomes not-clear.

## Map (8x8, x=column y=row, 0-indexed)

```
WWWWWWWW
WA    AW
W    B W
WB C   W
W   W  W
W      W
WGGG   W
WWWWWWWW
```
W=wall, A=agent, B=small box, C=heavy box, G=goal, space=empty.
Internal wall at (4,4). All border cells are walls.

## Naming

- Types: agent, location, box, heavybox
- Locations: `loc_{x}_{y}` (e.g. column 3 row 2 → loc_3_2)
- Agents: agent_0 at (1,1), agent_1 at (6,1)
- Boxes: box_0 at (5,2), box_1 at (1,3)
- Heavy box: hbx_0 at (3,3)
- Goals: (1,6), (2,6), (3,6)

## Goal State

```
(and (box-at box_0 loc_2_6) (box-at box_1 loc_1_6) (heavybox-at hbx_0 loc_3_6))
```

## Output

Two ```pddl code blocks: first domain.pddl, then problem.pddl.

Problem file must include:
- ALL 35 non-wall location objects (6x6 interior minus wall at (4,4))
- ALL adjacency pairs (bidirectional adj + directional adj-up/down/left/right)
- ALL in-line triples in BOTH orderings (left-to-right AND right-to-left, etc.)
- ALL clear predicates for cells without a box or heavybox (agent cells ARE clear)
- 3 goal-cell predicates
"""


# ---------------------------------------------------------------------------
# LLM-generated PDDL content (paste output from running LLM_PROMPT)
# ---------------------------------------------------------------------------
DOMAIN_PDDL = """
(define (domain box-pushing)
  (:requirements :strips :typing :equality)
  (:types agent location box heavybox)
  (:predicates
    (agent-at ?a - agent ?l - location)
    (box-at ?b - box ?l - location)
    (heavybox-at ?h - heavybox ?l - location)
    (clear ?l - location)
    (adj ?l1 - location ?l2 - location)
    (adj-up ?l1 - location ?l2 - location)
    (adj-down ?l1 - location ?l2 - location)
    (adj-left ?l1 - location ?l2 - location)
    (adj-right ?l1 - location ?l2 - location)
    (in-line ?l1 - location ?l2 - location ?l3 - location)
    (goal ?l - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?bloc - location ?to - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (box-at ?b ?bloc)
      (adj ?from ?bloc)
      (adj ?bloc ?to)
      (in-line ?from ?bloc ?to)
      (clear ?to))
    :effect (and
      (agent-at ?a ?bloc) (not (agent-at ?a ?from))
      (box-at ?b ?to) (not (box-at ?b ?bloc))
      (clear ?bloc) (not (clear ?to)))
  )

  (:action push-heavy-up
    :parameters (?a1 - agent ?a2 - agent ?from - location ?bloc - location ?to - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from) (agent-at ?a2 ?from)
      (adj-up ?from ?bloc) (heavybox-at ?h ?bloc)
      (adj-up ?bloc ?to) (clear ?to))
    :effect (and
      (agent-at ?a1 ?bloc) (agent-at ?a2 ?bloc)
      (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
      (heavybox-at ?h ?to) (not (heavybox-at ?h ?bloc))
      (clear ?bloc) (not (clear ?to)))
  )

  (:action push-heavy-down
    :parameters (?a1 - agent ?a2 - agent ?from - location ?bloc - location ?to - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from) (agent-at ?a2 ?from)
      (adj-down ?from ?bloc) (heavybox-at ?h ?bloc)
      (adj-down ?bloc ?to) (clear ?to))
    :effect (and
      (agent-at ?a1 ?bloc) (agent-at ?a2 ?bloc)
      (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
      (heavybox-at ?h ?to) (not (heavybox-at ?h ?bloc))
      (clear ?bloc) (not (clear ?to)))
  )

  (:action push-heavy-left
    :parameters (?a1 - agent ?a2 - agent ?from - location ?bloc - location ?to - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from) (agent-at ?a2 ?from)
      (adj-left ?from ?bloc) (heavybox-at ?h ?bloc)
      (adj-left ?bloc ?to) (clear ?to))
    :effect (and
      (agent-at ?a1 ?bloc) (agent-at ?a2 ?bloc)
      (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
      (heavybox-at ?h ?to) (not (heavybox-at ?h ?bloc))
      (clear ?bloc) (not (clear ?to)))
  )

  (:action push-heavy-right
    :parameters (?a1 - agent ?a2 - agent ?from - location ?bloc - location ?to - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from) (agent-at ?a2 ?from)
      (adj-right ?from ?bloc) (heavybox-at ?h ?bloc)
      (adj-right ?bloc ?to) (clear ?to))
    :effect (and
      (agent-at ?a1 ?bloc) (agent-at ?a2 ?bloc)
      (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
      (heavybox-at ?h ?to) (not (heavybox-at ?h ?bloc))
      (clear ?bloc) (not (clear ?to)))
  )
)"""

PROBLEM_PDDL = """
(define (problem box-pushing-1)
  (:domain box-pushing)
  (:objects
    agent_0 agent_1 - agent
    loc_1_1 loc_2_1 loc_3_1 loc_4_1 loc_5_1 loc_6_1 loc_1_2 loc_2_2 loc_3_2 loc_4_2 loc_5_2 loc_6_2 loc_1_3 loc_2_3 loc_3_3 loc_4_3 loc_5_3 loc_6_3 loc_1_4 loc_2_4 loc_3_4 loc_5_4 loc_6_4 loc_1_5 loc_2_5 loc_3_5 loc_4_5 loc_5_5 loc_6_5 loc_1_6 loc_2_6 loc_3_6 loc_4_6 loc_5_6 loc_6_6 - location
    box_0 box_1 - box
    hbx_0 - heavybox
  )
  (:init
    ;; Agent positions
    (agent-at agent_0 loc_1_1)
    (agent-at agent_1 loc_6_1)
    ;; Box positions
    (box-at box_0 loc_5_2)
    (box-at box_1 loc_1_3)
    ;; Heavybox positions
    (heavybox-at hbx_0 loc_3_3)
    ;; Goal cells
    (goal loc_1_6)
    (goal loc_2_6)
    (goal loc_3_6)
    ;; Clear cells (32 cells)
    (clear loc_1_1)
    (clear loc_2_1)
    (clear loc_3_1)
    (clear loc_4_1)
    (clear loc_5_1)
    (clear loc_6_1)
    (clear loc_1_2)
    (clear loc_2_2)
    (clear loc_3_2)
    (clear loc_4_2)
    (clear loc_6_2)
    (clear loc_2_3)
    (clear loc_4_3)
    (clear loc_5_3)
    (clear loc_6_3)
    (clear loc_1_4)
    (clear loc_2_4)
    (clear loc_3_4)
    (clear loc_5_4)
    (clear loc_6_4)
    (clear loc_1_5)
    (clear loc_2_5)
    (clear loc_3_5)
    (clear loc_4_5)
    (clear loc_5_5)
    (clear loc_6_5)
    (clear loc_1_6)
    (clear loc_2_6)
    (clear loc_3_6)
    (clear loc_4_6)
    (clear loc_5_6)
    (clear loc_6_6)
    ;; Bidirectional adjacency (112 facts)
    (adj loc_1_1 loc_1_2)
    (adj loc_1_1 loc_2_1)
    (adj loc_1_2 loc_1_1)
    (adj loc_1_2 loc_1_3)
    (adj loc_1_2 loc_2_2)
    (adj loc_1_3 loc_1_2)
    (adj loc_1_3 loc_1_4)
    (adj loc_1_3 loc_2_3)
    (adj loc_1_4 loc_1_3)
    (adj loc_1_4 loc_1_5)
    (adj loc_1_4 loc_2_4)
    (adj loc_1_5 loc_1_4)
    (adj loc_1_5 loc_1_6)
    (adj loc_1_5 loc_2_5)
    (adj loc_1_6 loc_1_5)
    (adj loc_1_6 loc_2_6)
    (adj loc_2_1 loc_1_1)
    (adj loc_2_1 loc_2_2)
    (adj loc_2_1 loc_3_1)
    (adj loc_2_2 loc_1_2)
    (adj loc_2_2 loc_2_1)
    (adj loc_2_2 loc_2_3)
    (adj loc_2_2 loc_3_2)
    (adj loc_2_3 loc_1_3)
    (adj loc_2_3 loc_2_2)
    (adj loc_2_3 loc_2_4)
    (adj loc_2_3 loc_3_3)
    (adj loc_2_4 loc_1_4)
    (adj loc_2_4 loc_2_3)
    (adj loc_2_4 loc_2_5)
    (adj loc_2_4 loc_3_4)
    (adj loc_2_5 loc_1_5)
    (adj loc_2_5 loc_2_4)
    (adj loc_2_5 loc_2_6)
    (adj loc_2_5 loc_3_5)
    (adj loc_2_6 loc_1_6)
    (adj loc_2_6 loc_2_5)
    (adj loc_2_6 loc_3_6)
    (adj loc_3_1 loc_2_1)
    (adj loc_3_1 loc_3_2)
    (adj loc_3_1 loc_4_1)
    (adj loc_3_2 loc_2_2)
    (adj loc_3_2 loc_3_1)
    (adj loc_3_2 loc_3_3)
    (adj loc_3_2 loc_4_2)
    (adj loc_3_3 loc_2_3)
    (adj loc_3_3 loc_3_2)
    (adj loc_3_3 loc_3_4)
    (adj loc_3_3 loc_4_3)
    (adj loc_3_4 loc_2_4)
    (adj loc_3_4 loc_3_3)
    (adj loc_3_4 loc_3_5)
    (adj loc_3_5 loc_2_5)
    (adj loc_3_5 loc_3_4)
    (adj loc_3_5 loc_3_6)
    (adj loc_3_5 loc_4_5)
    (adj loc_3_6 loc_2_6)
    (adj loc_3_6 loc_3_5)
    (adj loc_3_6 loc_4_6)
    (adj loc_4_1 loc_3_1)
    (adj loc_4_1 loc_4_2)
    (adj loc_4_1 loc_5_1)
    (adj loc_4_2 loc_3_2)
    (adj loc_4_2 loc_4_1)
    (adj loc_4_2 loc_4_3)
    (adj loc_4_2 loc_5_2)
    (adj loc_4_3 loc_3_3)
    (adj loc_4_3 loc_4_2)
    (adj loc_4_3 loc_5_3)
    (adj loc_4_5 loc_3_5)
    (adj loc_4_5 loc_4_6)
    (adj loc_4_5 loc_5_5)
    (adj loc_4_6 loc_3_6)
    (adj loc_4_6 loc_4_5)
    (adj loc_4_6 loc_5_6)
    (adj loc_5_1 loc_4_1)
    (adj loc_5_1 loc_5_2)
    (adj loc_5_1 loc_6_1)
    (adj loc_5_2 loc_4_2)
    (adj loc_5_2 loc_5_1)
    (adj loc_5_2 loc_5_3)
    (adj loc_5_2 loc_6_2)
    (adj loc_5_3 loc_4_3)
    (adj loc_5_3 loc_5_2)
    (adj loc_5_3 loc_5_4)
    (adj loc_5_3 loc_6_3)
    (adj loc_5_4 loc_5_3)
    (adj loc_5_4 loc_5_5)
    (adj loc_5_4 loc_6_4)
    (adj loc_5_5 loc_4_5)
    (adj loc_5_5 loc_5_4)
    (adj loc_5_5 loc_5_6)
    (adj loc_5_5 loc_6_5)
    (adj loc_5_6 loc_4_6)
    (adj loc_5_6 loc_5_5)
    (adj loc_5_6 loc_6_6)
    (adj loc_6_1 loc_5_1)
    (adj loc_6_1 loc_6_2)
    (adj loc_6_2 loc_5_2)
    (adj loc_6_2 loc_6_1)
    (adj loc_6_2 loc_6_3)
    (adj loc_6_3 loc_5_3)
    (adj loc_6_3 loc_6_2)
    (adj loc_6_3 loc_6_4)
    (adj loc_6_4 loc_5_4)
    (adj loc_6_4 loc_6_3)
    (adj loc_6_4 loc_6_5)
    (adj loc_6_5 loc_5_5)
    (adj loc_6_5 loc_6_4)
    (adj loc_6_5 loc_6_6)
    (adj loc_6_6 loc_5_6)
    (adj loc_6_6 loc_6_5)
    ;; adj-up (28 facts)
    (adj-up loc_1_2 loc_1_1)
    (adj-up loc_1_3 loc_1_2)
    (adj-up loc_1_4 loc_1_3)
    (adj-up loc_1_5 loc_1_4)
    (adj-up loc_1_6 loc_1_5)
    (adj-up loc_2_2 loc_2_1)
    (adj-up loc_2_3 loc_2_2)
    (adj-up loc_2_4 loc_2_3)
    (adj-up loc_2_5 loc_2_4)
    (adj-up loc_2_6 loc_2_5)
    (adj-up loc_3_2 loc_3_1)
    (adj-up loc_3_3 loc_3_2)
    (adj-up loc_3_4 loc_3_3)
    (adj-up loc_3_5 loc_3_4)
    (adj-up loc_3_6 loc_3_5)
    (adj-up loc_4_2 loc_4_1)
    (adj-up loc_4_3 loc_4_2)
    (adj-up loc_4_6 loc_4_5)
    (adj-up loc_5_2 loc_5_1)
    (adj-up loc_5_3 loc_5_2)
    (adj-up loc_5_4 loc_5_3)
    (adj-up loc_5_5 loc_5_4)
    (adj-up loc_5_6 loc_5_5)
    (adj-up loc_6_2 loc_6_1)
    (adj-up loc_6_3 loc_6_2)
    (adj-up loc_6_4 loc_6_3)
    (adj-up loc_6_5 loc_6_4)
    (adj-up loc_6_6 loc_6_5)
    ;; adj-down (28 facts)
    (adj-down loc_1_1 loc_1_2)
    (adj-down loc_1_2 loc_1_3)
    (adj-down loc_1_3 loc_1_4)
    (adj-down loc_1_4 loc_1_5)
    (adj-down loc_1_5 loc_1_6)
    (adj-down loc_2_1 loc_2_2)
    (adj-down loc_2_2 loc_2_3)
    (adj-down loc_2_3 loc_2_4)
    (adj-down loc_2_4 loc_2_5)
    (adj-down loc_2_5 loc_2_6)
    (adj-down loc_3_1 loc_3_2)
    (adj-down loc_3_2 loc_3_3)
    (adj-down loc_3_3 loc_3_4)
    (adj-down loc_3_4 loc_3_5)
    (adj-down loc_3_5 loc_3_6)
    (adj-down loc_4_1 loc_4_2)
    (adj-down loc_4_2 loc_4_3)
    (adj-down loc_4_5 loc_4_6)
    (adj-down loc_5_1 loc_5_2)
    (adj-down loc_5_2 loc_5_3)
    (adj-down loc_5_3 loc_5_4)
    (adj-down loc_5_4 loc_5_5)
    (adj-down loc_5_5 loc_5_6)
    (adj-down loc_6_1 loc_6_2)
    (adj-down loc_6_2 loc_6_3)
    (adj-down loc_6_3 loc_6_4)
    (adj-down loc_6_4 loc_6_5)
    (adj-down loc_6_5 loc_6_6)
    ;; adj-left (28 facts)
    (adj-left loc_2_1 loc_1_1)
    (adj-left loc_2_2 loc_1_2)
    (adj-left loc_2_3 loc_1_3)
    (adj-left loc_2_4 loc_1_4)
    (adj-left loc_2_5 loc_1_5)
    (adj-left loc_2_6 loc_1_6)
    (adj-left loc_3_1 loc_2_1)
    (adj-left loc_3_2 loc_2_2)
    (adj-left loc_3_3 loc_2_3)
    (adj-left loc_3_4 loc_2_4)
    (adj-left loc_3_5 loc_2_5)
    (adj-left loc_3_6 loc_2_6)
    (adj-left loc_4_1 loc_3_1)
    (adj-left loc_4_2 loc_3_2)
    (adj-left loc_4_3 loc_3_3)
    (adj-left loc_4_5 loc_3_5)
    (adj-left loc_4_6 loc_3_6)
    (adj-left loc_5_1 loc_4_1)
    (adj-left loc_5_2 loc_4_2)
    (adj-left loc_5_3 loc_4_3)
    (adj-left loc_5_5 loc_4_5)
    (adj-left loc_5_6 loc_4_6)
    (adj-left loc_6_1 loc_5_1)
    (adj-left loc_6_2 loc_5_2)
    (adj-left loc_6_3 loc_5_3)
    (adj-left loc_6_4 loc_5_4)
    (adj-left loc_6_5 loc_5_5)
    (adj-left loc_6_6 loc_5_6)
    ;; adj-right (28 facts)
    (adj-right loc_1_1 loc_2_1)
    (adj-right loc_1_2 loc_2_2)
    (adj-right loc_1_3 loc_2_3)
    (adj-right loc_1_4 loc_2_4)
    (adj-right loc_1_5 loc_2_5)
    (adj-right loc_1_6 loc_2_6)
    (adj-right loc_2_1 loc_3_1)
    (adj-right loc_2_2 loc_3_2)
    (adj-right loc_2_3 loc_3_3)
    (adj-right loc_2_4 loc_3_4)
    (adj-right loc_2_5 loc_3_5)
    (adj-right loc_2_6 loc_3_6)
    (adj-right loc_3_1 loc_4_1)
    (adj-right loc_3_2 loc_4_2)
    (adj-right loc_3_3 loc_4_3)
    (adj-right loc_3_5 loc_4_5)
    (adj-right loc_3_6 loc_4_6)
    (adj-right loc_4_1 loc_5_1)
    (adj-right loc_4_2 loc_5_2)
    (adj-right loc_4_3 loc_5_3)
    (adj-right loc_4_5 loc_5_5)
    (adj-right loc_4_6 loc_5_6)
    (adj-right loc_5_1 loc_6_1)
    (adj-right loc_5_2 loc_6_2)
    (adj-right loc_5_3 loc_6_3)
    (adj-right loc_5_4 loc_6_4)
    (adj-right loc_5_5 loc_6_5)
    (adj-right loc_5_6 loc_6_6)
    ;; In-line triples (84 facts)
    (in-line loc_1_1 loc_2_1 loc_3_1)
    (in-line loc_3_1 loc_2_1 loc_1_1)
    (in-line loc_2_1 loc_3_1 loc_4_1)
    (in-line loc_4_1 loc_3_1 loc_2_1)
    (in-line loc_3_1 loc_4_1 loc_5_1)
    (in-line loc_5_1 loc_4_1 loc_3_1)
    (in-line loc_4_1 loc_5_1 loc_6_1)
    (in-line loc_6_1 loc_5_1 loc_4_1)
    (in-line loc_1_2 loc_2_2 loc_3_2)
    (in-line loc_3_2 loc_2_2 loc_1_2)
    (in-line loc_2_2 loc_3_2 loc_4_2)
    (in-line loc_4_2 loc_3_2 loc_2_2)
    (in-line loc_3_2 loc_4_2 loc_5_2)
    (in-line loc_5_2 loc_4_2 loc_3_2)
    (in-line loc_4_2 loc_5_2 loc_6_2)
    (in-line loc_6_2 loc_5_2 loc_4_2)
    (in-line loc_1_3 loc_2_3 loc_3_3)
    (in-line loc_3_3 loc_2_3 loc_1_3)
    (in-line loc_2_3 loc_3_3 loc_4_3)
    (in-line loc_4_3 loc_3_3 loc_2_3)
    (in-line loc_3_3 loc_4_3 loc_5_3)
    (in-line loc_5_3 loc_4_3 loc_3_3)
    (in-line loc_4_3 loc_5_3 loc_6_3)
    (in-line loc_6_3 loc_5_3 loc_4_3)
    (in-line loc_1_4 loc_2_4 loc_3_4)
    (in-line loc_3_4 loc_2_4 loc_1_4)
    (in-line loc_1_5 loc_2_5 loc_3_5)
    (in-line loc_3_5 loc_2_5 loc_1_5)
    (in-line loc_2_5 loc_3_5 loc_4_5)
    (in-line loc_4_5 loc_3_5 loc_2_5)
    (in-line loc_3_5 loc_4_5 loc_5_5)
    (in-line loc_5_5 loc_4_5 loc_3_5)
    (in-line loc_4_5 loc_5_5 loc_6_5)
    (in-line loc_6_5 loc_5_5 loc_4_5)
    (in-line loc_1_6 loc_2_6 loc_3_6)
    (in-line loc_3_6 loc_2_6 loc_1_6)
    (in-line loc_2_6 loc_3_6 loc_4_6)
    (in-line loc_4_6 loc_3_6 loc_2_6)
    (in-line loc_3_6 loc_4_6 loc_5_6)
    (in-line loc_5_6 loc_4_6 loc_3_6)
    (in-line loc_4_6 loc_5_6 loc_6_6)
    (in-line loc_6_6 loc_5_6 loc_4_6)
    (in-line loc_1_1 loc_1_2 loc_1_3)
    (in-line loc_1_3 loc_1_2 loc_1_1)
    (in-line loc_1_2 loc_1_3 loc_1_4)
    (in-line loc_1_4 loc_1_3 loc_1_2)
    (in-line loc_1_3 loc_1_4 loc_1_5)
    (in-line loc_1_5 loc_1_4 loc_1_3)
    (in-line loc_1_4 loc_1_5 loc_1_6)
    (in-line loc_1_6 loc_1_5 loc_1_4)
    (in-line loc_2_1 loc_2_2 loc_2_3)
    (in-line loc_2_3 loc_2_2 loc_2_1)
    (in-line loc_2_2 loc_2_3 loc_2_4)
    (in-line loc_2_4 loc_2_3 loc_2_2)
    (in-line loc_2_3 loc_2_4 loc_2_5)
    (in-line loc_2_5 loc_2_4 loc_2_3)
    (in-line loc_2_4 loc_2_5 loc_2_6)
    (in-line loc_2_6 loc_2_5 loc_2_4)
    (in-line loc_3_1 loc_3_2 loc_3_3)
    (in-line loc_3_3 loc_3_2 loc_3_1)
    (in-line loc_3_2 loc_3_3 loc_3_4)
    (in-line loc_3_4 loc_3_3 loc_3_2)
    (in-line loc_3_3 loc_3_4 loc_3_5)
    (in-line loc_3_5 loc_3_4 loc_3_3)
    (in-line loc_3_4 loc_3_5 loc_3_6)
    (in-line loc_3_6 loc_3_5 loc_3_4)
    (in-line loc_4_1 loc_4_2 loc_4_3)
    (in-line loc_4_3 loc_4_2 loc_4_1)
    (in-line loc_5_1 loc_5_2 loc_5_3)
    (in-line loc_5_3 loc_5_2 loc_5_1)
    (in-line loc_5_2 loc_5_3 loc_5_4)
    (in-line loc_5_4 loc_5_3 loc_5_2)
    (in-line loc_5_3 loc_5_4 loc_5_5)
    (in-line loc_5_5 loc_5_4 loc_5_3)
    (in-line loc_5_4 loc_5_5 loc_5_6)
    (in-line loc_5_6 loc_5_5 loc_5_4)
    (in-line loc_6_1 loc_6_2 loc_6_3)
    (in-line loc_6_3 loc_6_2 loc_6_1)
    (in-line loc_6_2 loc_6_3 loc_6_4)
    (in-line loc_6_4 loc_6_3 loc_6_2)
    (in-line loc_6_3 loc_6_4 loc_6_5)
    (in-line loc_6_5 loc_6_4 loc_6_3)
    (in-line loc_6_4 loc_6_5 loc_6_6)
    (in-line loc_6_6 loc_6_5 loc_6_4)
  )
  (:goal (and
    (box-at box_0 loc_2_6)
    (box-at box_1 loc_1_6)
    (heavybox-at hbx_0 loc_3_6)
  ))
)
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    if not DOMAIN_PDDL or not PROBLEM_PDDL:
        print("ERROR: DOMAIN_PDDL and PROBLEM_PDDL must be populated.")
        sys.exit(1)

    # Write PDDL files
    os.makedirs(PDDL_DIR, exist_ok=True)
    for path, content in [(DOMAIN_FILE, DOMAIN_PDDL), (PROBLEM_FILE, PROBLEM_PDDL)]:
        with open(path, "w") as f:
            f.write(content)

    # Round-trip verify
    ascii_map = pddl_to_ascii_map(PROBLEM_FILE)
    print("Map:")
    for row in ascii_map:
        print(f"  {row}")

    # Solve
    plan = solve_pddl(DOMAIN_FILE, PROBLEM_FILE)
    if not plan:
        print("No plan found!")
        sys.exit(1)
    print(f"\nPlan ({len(plan.actions)} steps):")
    for i, action in enumerate(plan.actions):
        print(f"  {i + 1}. {action}")

    # Visualize
    visualize_pddl_plan(ascii_map, DOMAIN_FILE, PROBLEM_FILE)


if __name__ == "__main__":
    run_pipeline()
