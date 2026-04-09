(define (domain box-pushing)
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
)