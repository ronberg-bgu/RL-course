(define (domain box-push)
  (:requirements :strips :typing :equality)
  (:types agent location box heavybox direction)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (heavybox-at ?h - heavybox ?loc - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (move-dir ?l1 - location ?l2 - location ?d - direction)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and
        (agent-at ?a ?from)
        (adj ?from ?to)
        (clear ?to)
    )
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)))
  )

  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box ?d - direction)
    :precondition (and
        (agent-at ?a ?from)
        (move-dir ?from ?boxloc ?d)
        (box-at ?b ?boxloc)
        (move-dir ?boxloc ?toloc ?d)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a ?boxloc)
        (not (agent-at ?a ?from))
        (box-at ?b ?toloc)
        (not (box-at ?b ?boxloc))
        (not (clear ?toloc))
        (clear ?boxloc)
    )
  )

  (:action push-heavy
    :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox ?d - direction)
    :precondition (and
        (not (= ?a1 ?a2))
        (agent-at ?a1 ?from)
        (agent-at ?a2 ?from)
        (move-dir ?from ?boxloc ?d)
        (heavybox-at ?h ?boxloc)
        (move-dir ?boxloc ?toloc ?d)
        (clear ?toloc)
    )
    :effect (and
        (agent-at ?a1 ?boxloc)
        (agent-at ?a2 ?boxloc)
        (not (agent-at ?a1 ?from))
        (not (agent-at ?a2 ?from))
        (heavybox-at ?h ?toloc)
        (not (heavybox-at ?h ?boxloc))
        (not (clear ?toloc))
        (clear ?boxloc)
    )
  )
)
