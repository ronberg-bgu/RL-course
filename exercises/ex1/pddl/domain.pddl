(define (domain box-push)
  (:requirements :strips :typing :equality)
  (:types agent location box heavybox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?b - heavybox ?loc - location)
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
    )
  )

  (:action push-small-up
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?other - agent)
    :precondition (and
      (not (= ?a ?other))
      (agent-at ?a ?from)
      (adj-up ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-up ?boxloc ?toloc)
      (clear ?toloc)
      (not (agent-at ?other ?toloc))
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from))
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-small-down
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?other - agent)
    :precondition (and
      (not (= ?a ?other))
      (agent-at ?a ?from)
      (adj-down ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-down ?boxloc ?toloc)
      (clear ?toloc)
      (not (agent-at ?other ?toloc))
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from))
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-small-left
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?other - agent)
    :precondition (and
      (not (= ?a ?other))
      (agent-at ?a ?from)
      (adj-left ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-left ?boxloc ?toloc)
      (clear ?toloc)
      (not (agent-at ?other ?toloc))
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from))
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-small-right
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?other - agent)
    :precondition (and
      (not (= ?a ?other))
      (agent-at ?a ?from)
      (adj-right ?from ?boxloc)
      (box-at ?b ?boxloc)
      (adj-right ?boxloc ?toloc)
      (clear ?toloc)
      (not (agent-at ?other ?toloc))
    )
    :effect (and
      (agent-at ?a ?boxloc) (not (agent-at ?a ?from))
      (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-heavy-up
    :parameters (?a1 ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?b - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (adj-up ?from ?boxloc)
      (heavybox-at ?b ?boxloc)
      (adj-up ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc) (not (agent-at ?a1 ?from))
      (agent-at ?a2 ?boxloc) (not (agent-at ?a2 ?from))
      (heavybox-at ?b ?toloc) (not (heavybox-at ?b ?boxloc))
      (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-heavy-down
    :parameters (?a1 ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?b - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (adj-down ?from ?boxloc)
      (heavybox-at ?b ?boxloc)
      (adj-down ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc) (not (agent-at ?a1 ?from))
      (agent-at ?a2 ?boxloc) (not (agent-at ?a2 ?from))
      (heavybox-at ?b ?toloc) (not (heavybox-at ?b ?boxloc))
      (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-heavy-right
    :parameters (?a1 ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?b - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (adj-right ?from ?boxloc)
      (heavybox-at ?b ?boxloc)
      (adj-right ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc) (not (agent-at ?a1 ?from))
      (agent-at ?a2 ?boxloc) (not (agent-at ?a2 ?from))
      (heavybox-at ?b ?toloc) (not (heavybox-at ?b ?boxloc))
      (not (clear ?toloc))
      (clear ?boxloc)
    )
  )

  (:action push-heavy-left
    :parameters (?a1 ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?b - heavybox)
    :precondition (and
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?from)
      (agent-at ?a2 ?from)
      (adj-left ?from ?boxloc)
      (heavybox-at ?b ?boxloc)
      (adj-left ?boxloc ?toloc)
      (clear ?toloc)
    )
    :effect (and
      (agent-at ?a1 ?boxloc) (not (agent-at ?a1 ?from))
      (agent-at ?a2 ?boxloc) (not (agent-at ?a2 ?from))
      (heavybox-at ?b ?toloc) (not (heavybox-at ?b ?boxloc))
      (not (clear ?toloc))
      (clear ?boxloc)
    )
  )
)
