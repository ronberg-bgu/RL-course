from visualize_plan import visualize_pddl_plan
from pddl_to_map import parse_pddl_to_map   # your function name may differ
import sys
sys.stdout.reconfigure(encoding='utf-8')
prompt_string = """
The goal is to generate a PDDL for a box pushing problem:
  There are 2 actors and three boxes - one of them is a big box that requires two actors to push it 
 Types: agent location  box heavybox;

Predicates: agent-at box-at heavybox-at
example syntax: agent_0, agent_1 loc_x_y box_0, box_1 hbox_0 
 Operations and parameters: 
 move : ?a - agent ?from - location ?to - location 
 push-small: ?a - agent ?from - location  ?boxloc - location  ?toloc - location  ?b - box 
push-heavy-up: ?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location  ?b - heavybox  
 push-heavy-down :same as above  
 push-heavy-left : same as above  
 push-heavy-right : same as above 
the two coordinates in push-big-x are used for box location and agent location, since two agents need to push from the exact same point adjacent to the heavy box; 
additional predicates that can be used for better effect:
north south east west - for heavy box pushing;
adj, clear - for marking cells;
 Example Goal: (:goal (and     (box-at box_0 loc_3_5)     (box-at box_1 loc_4_5) ))
 Big box is 1x1 in size and requires two agents to push at the same time 

multiple agents can occupy the same space;
make sure to avoid diagonal pushing for heavy box;
make sure to avoid pushing into boxes (or agents);

Please give a COMPLETE domain PDDL, problem PDDL is not necessary for now
"""

domain_string = """(define (domain box-pushing)
  (:requirements :strips :typing :equality)

  (:types 
    agent 
    location 
    box 
    heavybox
    direction
  )

  (:constants
    dir-up dir-down dir-left dir-right - direction
  )

  (:predicates
    (agent-at ?a - agent ?l - location)
    (box-at ?b - box ?l - location)
    (heavybox-at ?b - heavybox ?l - location)

    (adj-dir ?from ?to - location ?d - direction)

    ;; only boxes affect this
    (clear ?l - location)
  )

  ;; MOVE
  (:action move
    :parameters (?a - agent ?from ?to - location ?d - direction)
    :precondition (and
      (agent-at ?a ?from)
      (adj-dir ?from ?to ?d)
      (clear ?to)
    )
    :effect (and
      (not (agent-at ?a ?from))
      (agent-at ?a ?to)
    )
  )

  ;; PUSH SMALL
  (:action push-small
    :parameters (?a - agent ?from ?boxloc ?toloc - location ?b - box ?d - direction)
    :precondition (and
      (agent-at ?a ?from)
      (box-at ?b ?boxloc)

      (adj-dir ?from ?boxloc ?d)
      (adj-dir ?boxloc ?toloc ?d)

      (clear ?toloc)
    )
    :effect (and
      (not (agent-at ?a ?from))
      (agent-at ?a ?boxloc)

      (not (box-at ?b ?boxloc))
      (box-at ?b ?toloc)

      (clear ?boxloc)
      (not (clear ?toloc))
    )
  )

  ;; PUSH HEAVY
  (:action push-heavy
    :parameters (?a1 ?a2 - agent ?from ?boxloc ?toloc - location ?b - heavybox ?d - direction)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)

      (heavybox-at ?b ?boxloc)

      (adj-dir ?from ?boxloc ?d)
      (adj-dir ?boxloc ?toloc ?d)

      (clear ?toloc)
    )
    :effect (and
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)

      (not (heavybox-at ?b ?boxloc))
      (heavybox-at ?b ?toloc)

      (clear ?boxloc)
      (not (clear ?toloc))
    )
  )
)"""

problem_string = """(define (problem box-pushing-problem)
  (:domain box-pushing)

  (:objects
    agent_0 agent_1 - agent
    box_0 box_1 - box
    hbox_0 - heavybox

    loc_1_1 loc_2_1 loc_3_1 loc_4_1 loc_5_1 loc_6_1
    loc_1_2 loc_2_2 loc_3_2 loc_4_2 loc_5_2 loc_6_2
    loc_1_3 loc_2_3 loc_3_3 loc_4_3 loc_5_3 loc_6_3
    loc_1_4 loc_2_4 loc_3_4 loc_4_4 loc_5_4 loc_6_4
    loc_1_5 loc_2_5 loc_3_5 loc_4_5 loc_5_5 loc_6_5
    - location
  )

  (:init
    ;; AGENTS
    (agent-at agent_0 loc_3_1)
    (agent-at agent_1 loc_4_1)

    ;; BOXES
    (box-at box_0 loc_2_2)
    (box-at box_1 loc_4_4)

    ;; HEAVY BOX
    (heavybox-at hbox_0 loc_4_2)

    ;; CLEAR (only where no boxes)
    (clear loc_1_1) (clear loc_2_1) (clear loc_3_1) (clear loc_4_1) (clear loc_5_1) (clear loc_6_1)

    (clear loc_1_2) (clear loc_3_2) (clear loc_5_2) (clear loc_6_2)

    (clear loc_1_3) (clear loc_2_3) (clear loc_3_3)
    (clear loc_4_3) (clear loc_5_3) (clear loc_6_3)

    (clear loc_1_4) (clear loc_2_4) (clear loc_3_4)
    (clear loc_5_4) (clear loc_6_4)

    (clear loc_1_5) (clear loc_2_5) (clear loc_3_5)
    (clear loc_4_5) (clear loc_5_5) (clear loc_6_5)

    ;; ADJACENCY (dir-right / dir-left)
    (adj-dir loc_1_1 loc_2_1 dir-right) (adj-dir loc_2_1 loc_1_1 dir-left)
    (adj-dir loc_2_1 loc_3_1 dir-right) (adj-dir loc_3_1 loc_2_1 dir-left)
    (adj-dir loc_3_1 loc_4_1 dir-right) (adj-dir loc_4_1 loc_3_1 dir-left)
    (adj-dir loc_4_1 loc_5_1 dir-right) (adj-dir loc_5_1 loc_4_1 dir-left)
    (adj-dir loc_5_1 loc_6_1 dir-right) (adj-dir loc_6_1 loc_5_1 dir-left)

    (adj-dir loc_1_2 loc_2_2 dir-right) (adj-dir loc_2_2 loc_1_2 dir-left)
    (adj-dir loc_2_2 loc_3_2 dir-right) (adj-dir loc_3_2 loc_2_2 dir-left)
    (adj-dir loc_3_2 loc_4_2 dir-right) (adj-dir loc_4_2 loc_3_2 dir-left)
    (adj-dir loc_4_2 loc_5_2 dir-right) (adj-dir loc_5_2 loc_4_2 dir-left)
    (adj-dir loc_5_2 loc_6_2 dir-right) (adj-dir loc_6_2 loc_5_2 dir-left)

    (adj-dir loc_1_3 loc_2_3 dir-right) (adj-dir loc_2_3 loc_1_3 dir-left)
    (adj-dir loc_2_3 loc_3_3 dir-right) (adj-dir loc_3_3 loc_2_3 dir-left)
    (adj-dir loc_3_3 loc_4_3 dir-right) (adj-dir loc_4_3 loc_3_3 dir-left)
    (adj-dir loc_4_3 loc_5_3 dir-right) (adj-dir loc_5_3 loc_4_3 dir-left)
    (adj-dir loc_5_3 loc_6_3 dir-right) (adj-dir loc_6_3 loc_5_3 dir-left)

    (adj-dir loc_1_4 loc_2_4 dir-right) (adj-dir loc_2_4 loc_1_4 dir-left)
    (adj-dir loc_2_4 loc_3_4 dir-right) (adj-dir loc_3_4 loc_2_4 dir-left)
    (adj-dir loc_3_4 loc_4_4 dir-right) (adj-dir loc_4_4 loc_3_4 dir-left)
    (adj-dir loc_4_4 loc_5_4 dir-right) (adj-dir loc_5_4 loc_4_4 dir-left)
    (adj-dir loc_5_4 loc_6_4 dir-right) (adj-dir loc_6_4 loc_5_4 dir-left)

    (adj-dir loc_1_5 loc_2_5 dir-right) (adj-dir loc_2_5 loc_1_5 dir-left)
    (adj-dir loc_2_5 loc_3_5 dir-right) (adj-dir loc_3_5 loc_2_5 dir-left)
    (adj-dir loc_3_5 loc_4_5 dir-right) (adj-dir loc_4_5 loc_3_5 dir-left)
    (adj-dir loc_4_5 loc_5_5 dir-right) (adj-dir loc_5_5 loc_4_5 dir-left)
    (adj-dir loc_5_5 loc_6_5 dir-right) (adj-dir loc_6_5 loc_5_5 dir-left)

    ;; UP / DOWN
    (adj-dir loc_1_1 loc_1_2 dir-up) (adj-dir loc_1_2 loc_1_1 dir-down)
    (adj-dir loc_1_2 loc_1_3 dir-up) (adj-dir loc_1_3 loc_1_2 dir-down)
    (adj-dir loc_1_3 loc_1_4 dir-up) (adj-dir loc_1_4 loc_1_3 dir-down)
    (adj-dir loc_1_4 loc_1_5 dir-up) (adj-dir loc_1_5 loc_1_4 dir-down)

    (adj-dir loc_2_1 loc_2_2 dir-up) (adj-dir loc_2_2 loc_2_1 dir-down)
    (adj-dir loc_2_2 loc_2_3 dir-up) (adj-dir loc_2_3 loc_2_2 dir-down)
    (adj-dir loc_2_3 loc_2_4 dir-up) (adj-dir loc_2_4 loc_2_3 dir-down)
    (adj-dir loc_2_4 loc_2_5 dir-up) (adj-dir loc_2_5 loc_2_4 dir-down)

    (adj-dir loc_3_1 loc_3_2 dir-up) (adj-dir loc_3_2 loc_3_1 dir-down)
    (adj-dir loc_3_2 loc_3_3 dir-up) (adj-dir loc_3_3 loc_3_2 dir-down)
    (adj-dir loc_3_3 loc_3_4 dir-up) (adj-dir loc_3_4 loc_3_3 dir-down)
    (adj-dir loc_3_4 loc_3_5 dir-up) (adj-dir loc_3_5 loc_3_4 dir-down)

    (adj-dir loc_4_1 loc_4_2 dir-up) (adj-dir loc_4_2 loc_4_1 dir-down)
    (adj-dir loc_4_2 loc_4_3 dir-up) (adj-dir loc_4_3 loc_4_2 dir-down)
    (adj-dir loc_4_3 loc_4_4 dir-up) (adj-dir loc_4_4 loc_4_3 dir-down)
    (adj-dir loc_4_4 loc_4_5 dir-up) (adj-dir loc_4_5 loc_4_4 dir-down)

    (adj-dir loc_5_1 loc_5_2 dir-up) (adj-dir loc_5_2 loc_5_1 dir-down)
    (adj-dir loc_5_2 loc_5_3 dir-up) (adj-dir loc_5_3 loc_5_2 dir-down)
    (adj-dir loc_5_3 loc_5_4 dir-up) (adj-dir loc_5_4 loc_5_3 dir-down)
    (adj-dir loc_5_4 loc_5_5 dir-up) (adj-dir loc_5_5 loc_5_4 dir-down)

    (adj-dir loc_6_1 loc_6_2 dir-up) (adj-dir loc_6_2 loc_6_1 dir-down)
    (adj-dir loc_6_2 loc_6_3 dir-up) (adj-dir loc_6_3 loc_6_2 dir-down)
    (adj-dir loc_6_3 loc_6_4 dir-up) (adj-dir loc_6_4 loc_6_3 dir-down)
    (adj-dir loc_6_4 loc_6_5 dir-up) (adj-dir loc_6_5 loc_6_4 dir-down)
  )

  (:goal
    (and
      (box-at box_0 loc_2_5)
      (box-at box_1 loc_4_5)
      (heavybox-at hbox_0 loc_6_5)
    )
  )
)"""




if __name__ == "__main__":
    with open("pddl/domain.pddl", "w") as f1:
        f1.write(domain_string)
    with open("pddl/problem.pddl", "w") as f2:
        f2.write(problem_string)
    ascii_map = parse_pddl_to_map("pddl/problem.pddl")
    visualize_pddl_plan(ascii_map, "pddl/domain.pddl", "pddl/problem.pddl")
