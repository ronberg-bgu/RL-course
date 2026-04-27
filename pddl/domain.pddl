(define (domain box-push)
  (:requirements :typing :equality :disjunctive-preconditions)
  (:types
    agent location box heavybox
  )

  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (left-of ?l1 - location ?l2 - location)
    (right-of ?l1 - location ?l2 - location)
    (up-of ?l1 - location ?l2 - location)
    (down-of ?l1 - location ?l2 - location)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and
      (agent-at ?a ?from)
      (adj ?from ?to)
      (clear ?to)
    )
    :effect (and
      (not (agent-at ?a ?from))
      (agent-at ?a ?to)
    )
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (box-at ?b ?boxloc)
      (clear ?toloc)
      (or
        (and (left-of ?from ?boxloc) (left-of ?boxloc ?toloc))
        (and (right-of ?from ?boxloc) (right-of ?boxloc ?toloc))
        (and (up-of ?from ?boxloc) (up-of ?boxloc ?toloc))
        (and (down-of ?from ?boxloc) (down-of ?boxloc ?toloc))
      )
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

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (heavybox-at ?h ?boxloc)
      (clear ?toloc)
      (or
        (and (left-of ?from ?boxloc) (left-of ?boxloc ?toloc))
        (and (right-of ?from ?boxloc) (right-of ?boxloc ?toloc))
        (and (up-of ?from ?boxloc) (up-of ?boxloc ?toloc))
        (and (down-of ?from ?boxloc) (down-of ?boxloc ?toloc))
      )
    )
    :effect (and
      (not (agent-at ?a1 ?from))
      (not (agent-at ?a2 ?from))
      (agent-at ?a1 ?boxloc)
      (agent-at ?a2 ?boxloc)
      (not (heavybox-at ?h ?boxloc))
      (heavybox-at ?h ?toloc)
      (clear ?boxloc)
      (not (clear ?toloc))
    )
  )
)