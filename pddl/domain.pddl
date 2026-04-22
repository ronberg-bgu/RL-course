(define (domain box-pushing)
    (:requirements :typing :equality)
    
    (:types
        agent box heavybox location - object
    )
    
    (:predicates
        (at-agent ?a - agent ?loc - location)
        (at-box ?b - box ?loc - location)
        (at-heavybox ?h - heavybox ?loc - location)
        
        ;; True if a tile has no box (regular or heavy) and is not an obstacle.
        (clear ?loc - location)
        
        ;; Basic adjacency for walking
        (adj ?l1 - location ?l2 - location)
        
        ;; Represents 3 tiles in a straight, contiguous line (from -> boxloc -> toloc)
        (in-line ?l1 - location ?l2 - location ?l3 - location)
    )

    ;; Agent moves from one tile to an adjacent tile if it's clear of boxes/obstacles.
    (:action move
        :parameters (?a - agent ?from - location ?to - location)
        :precondition (and 
            (at-agent ?a ?from)
            (adj ?from ?to)
            (clear ?to)
        )
        :effect (and 
            (not (at-agent ?a ?from))
            (at-agent ?a ?to)
        )
    )

    ;; A single agent pushes a small box into an empty, clear adjacent space in a straight line.
    (:action push-small
        :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
        :precondition (and 
            (at-agent ?a ?from)
            (at-box ?b ?boxloc)
            (in-line ?from ?boxloc ?toloc)
            (clear ?toloc)
        )
        :effect (and 
            (not (at-agent ?a ?from))
            (at-agent ?a ?boxloc)
            (not (at-box ?b ?boxloc))
            (at-box ?b ?toloc)
            (not (clear ?toloc))
            (clear ?boxloc)
        )
    )

    ;; Two agents on the same tile push a heavy box into an empty space in a straight line.
    (:action push-heavy
        :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
        :precondition (and 
            (not (= ?a1 ?a2))
            (at-agent ?a1 ?from)
            (at-agent ?a2 ?from)
            (at-heavybox ?h ?boxloc)
            (in-line ?from ?boxloc ?toloc)
            (clear ?toloc)
        )
        :effect (and 
            (not (at-agent ?a1 ?from))
            (not (at-agent ?a2 ?from))
            (at-agent ?a1 ?boxloc)
            (at-agent ?a2 ?boxloc)
            (not (at-heavybox ?h ?boxloc))
            (at-heavybox ?h ?toloc)
            (not (clear ?toloc))
            (clear ?boxloc)
        )
    )
)