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
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and
      (agent-at ?a ?to) (not (agent-at ?a ?from))
      (clear ?from) (not (clear ?to))
    )
  )

  (:action push-small-up
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (adj-up ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-up ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from)
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
    )
  )

  (:action push-small-down
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (adj-down ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-down ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from)
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
    )
  )

  (:action push-small-left
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (adj-left ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-left ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from)
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
    )
  )

  (:action push-small-right
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and
      (agent-at ?a ?from)
      (adj-right ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-right ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from)
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
    )
  )

  (:action push-big-up
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and
      (agent-at ?a1 ?from1) (adj-up ?from1 ?boxloc1)
      (agent-at ?a2 ?from2) (adj-up ?from2 ?boxloc2)
      (bigbox-at ?b ?boxloc1 ?boxloc2)
      (adj-right ?boxloc1 ?boxloc2)
      (adj-up ?boxloc1 ?toloc1) (clear ?toloc1)
      (adj-up ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and
      (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
      (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
      (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
      (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-down
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and
      (agent-at ?a1 ?from1) (adj-down ?from1 ?boxloc1)
      (agent-at ?a2 ?from2) (adj-down ?from2 ?boxloc2)
      (bigbox-at ?b ?boxloc1 ?boxloc2)
      (adj-right ?boxloc1 ?boxloc2)
      (adj-down ?boxloc1 ?toloc1) (clear ?toloc1)
      (adj-down ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and
      (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
      (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
      (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
      (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-right
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and
      (agent-at ?a1 ?from1) (adj-right ?from1 ?boxloc1)
      (agent-at ?a2 ?from2) (adj-right ?from2 ?boxloc2)
      (bigbox-at ?b ?boxloc1 ?boxloc2)
      (adj-down ?boxloc1 ?boxloc2)
      (adj-right ?boxloc1 ?toloc1) (clear ?toloc1)
      (adj-right ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and
      (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
      (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
      (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
      (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-left
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and
      (agent-at ?a1 ?from1) (adj-left ?from1 ?boxloc1)
      (agent-at ?a2 ?from2) (adj-left ?from2 ?boxloc2)
      (bigbox-at ?b ?boxloc1 ?boxloc2)
      (adj-down ?boxloc1 ?boxloc2)
      (adj-left ?boxloc1 ?toloc1) (clear ?toloc1)
      (adj-left ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and
      (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
      (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
      (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
      (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )
)
