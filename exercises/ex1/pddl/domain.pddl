(define (domain box-pushing)
    (:requirements :strips :typing :equality)
    (:types agent location box heavybox)

    (:predicates
        (agent-at ?ag - agent ?loc - location)
        (box-at ?b - box ?loc - location)
        (heavybox-at ?hb - heavybox ?loc - location)
        (clear ?loc - location)
        (adjacent ?loc1 - location ?loc2 - location)
        (straight ?loc1 - location ?loc2 - location ?loc3 - location)
    )

    ;; Agent movement
    (:action move
        :parameters (?ag - agent ?from - location ?to - location)
        :precondition (and
            (agent-at ?ag ?from)
            (adjacent ?from ?to)
            (clear ?to)
        )
        :effect (and
            (not (agent-at ?ag ?from))
            (agent-at ?ag ?to)
        )
    )

    ;; Push a regular box
    (:action push-small
        :parameters (?ag - agent ?from - location ?to - location ?b - box ?new_box_loc - location)
        :precondition (and
            (agent-at ?ag ?from)
            (box-at ?b ?to)
            (straight ?from ?to ?new_box_loc)
            (clear ?new_box_loc)
        )
        :effect (and
            (not (agent-at ?ag ?from))
            (agent-at ?ag ?to)
            (not (box-at ?b ?to))
            (box-at ?b ?new_box_loc)
            (clear ?to)
            (not (clear ?new_box_loc))
        )
    )

    ;; Push a heavy box with two agents
    (:action push-heavy
        :parameters (?ag1 - agent ?ag2 - agent ?from - location ?to - location ?hb - heavybox ?new_box_loc - location)
        :precondition (and
            (agent-at ?ag1 ?from)
            (agent-at ?ag2 ?from)
            (heavybox-at ?hb ?to)
            (straight ?from ?to ?new_box_loc)
            (clear ?new_box_loc)
            (not (= ?ag1 ?ag2)) ; Ensure two distinct agents
        )
        :effect (and
            (not (agent-at ?ag1 ?from))
            (not (agent-at ?ag2 ?from))
            (agent-at ?ag1 ?to)
            (agent-at ?ag2 ?to)
            (not (heavybox-at ?hb ?to))
            (heavybox-at ?hb ?new_box_loc)
            (clear ?to)
            (not (clear ?new_box_loc))
        )
    )
)