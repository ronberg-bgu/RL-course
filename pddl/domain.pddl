(define (domain box-push)
  (:requirements :strips :typing :equality :disjunctive-preconditions)

  (:types
    agent location box heavybox
  )

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
        (and (adj-up ?from ?boxloc)    (adj-up ?boxloc ?toloc))
        (and (adj-down ?from ?boxloc)  (adj-down ?boxloc ?toloc))
        (and (adj-left ?from ?boxloc)  (adj-left ?boxloc ?toloc))
        (and (adj-right ?from ?boxloc) (adj-right ?boxloc ?toloc))
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
        (and (adj-up ?from ?boxloc)    (adj-up ?boxloc ?toloc))
        (and (adj-down ?from ?boxloc)  (adj-down ?boxloc ?toloc))
        (and (adj-left ?from ?boxloc)  (adj-left ?boxloc ?toloc))
        (and (adj-right ?from ?boxloc) (adj-right ?boxloc ?toloc))
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
