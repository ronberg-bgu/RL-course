(define (domain box-push)
  (:requirements :strips :typing)
  (:types agent location box heavybox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (adj-left ?l1 - location ?l2 - location)
    (adj-right ?l1 - location ?l2 - location)
    (adj-up ?l1 - location ?l2 - location)
    (adj-down ?l1 - location ?l2 - location)
    (in-line ?a - location ?b - location ?c - location)
    (goal ?loc - location)
    (won )
  )

  ;; 1. move — agent moves to any adjacent clear cell
  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  ;; 2. push-small — one agent pushes a regular box (collinear only)
  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
        (agent-at ?a ?from)
        (box-at ?b ?boxloc)
        (adj ?from ?boxloc)
        (adj ?boxloc ?toloc)
        (in-line ?from ?boxloc ?toloc)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from)
        (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (clear ?boxloc) (not (clear ?toloc))
    )
  )

  ;; 3-6. push-heavy — both agents push heavy box (directional variants)
  (:action push-heavy-up
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from) (agent-at ?a2 ?from)
        (adj-up ?from ?boxloc)
        (heavybox-at ?h ?boxloc)
        (adj-up ?boxloc ?toloc)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
        (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
        (clear ?from)
        (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc)) (clear ?boxloc) (not (clear ?toloc))
    )
  )
  (:action push-heavy-down
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from) (agent-at ?a2 ?from)
        (adj-down ?from ?boxloc)
        (heavybox-at ?h ?boxloc)
        (adj-down ?boxloc ?toloc)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
        (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
        (clear ?from)
        (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc)) (clear ?boxloc) (not (clear ?toloc))
    )
  )
  (:action push-heavy-left
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from) (agent-at ?a2 ?from)
        (adj-left ?from ?boxloc)
        (heavybox-at ?h ?boxloc)
        (adj-left ?boxloc ?toloc)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
        (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
        (clear ?from)
        (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc)) (clear ?boxloc) (not (clear ?toloc))
    )
  )
  (:action push-heavy-right
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from) (agent-at ?a2 ?from)
        (adj-right ?from ?boxloc)
        (heavybox-at ?h ?boxloc)
        (adj-right ?boxloc ?toloc)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
        (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
        (clear ?from)
        (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc)) (clear ?boxloc) (not (clear ?toloc))
    )
  )

  (:action win-small
    :parameters (?b - box ?loc - location)
    :precondition (and (box-at ?b ?loc) (goal ?loc))
    :effect (won)
  )

  (:action win-heavy
    :parameters (?h - heavybox ?loc - location)
    :precondition (and (heavybox-at ?h ?loc) (goal ?loc))
    :effect (won)
  )
)
