(define (problem box-pushing-problem)
    (:domain box-pushing)
    (:objects
        agent_0 agent_1 - agent
        box_0 box_1 - box
        heavy_0 - heavybox
        loc_1_1 loc_1_2 loc_1_3 loc_1_4 loc_1_5
        loc_2_1 loc_2_2 loc_2_3 loc_2_4 loc_2_5
        loc_3_1 loc_3_2 loc_3_3 loc_3_4 loc_3_5
        loc_4_1 loc_4_2 loc_4_3 loc_4_4 loc_4_5
        loc_5_1 loc_5_2 loc_5_3 loc_5_4 loc_5_5 - location
    )

    (:init
        ;; Agent locations
        (agent-at agent_0 loc_5_1)
        (agent-at agent_1 loc_1_2)

        ;; Box locations
        (box-at box_0 loc_2_2)
        (box-at box_1 loc_3_3)
        (heavybox-at heavy_0 loc_2_3)

        ;; Clear locations (initially, all locations not occupied by a box/heavybox)
        (clear loc_1_1) (clear loc_1_2) (clear loc_1_3) (clear loc_1_4) (clear loc_1_5)
        (clear loc_2_1)                      (clear loc_2_4) (clear loc_2_5)
        (clear loc_3_1) (clear loc_3_2)                      (clear loc_3_4) (clear loc_3_5)
        (clear loc_4_1) (clear loc_4_2) (clear loc_4_3) (clear loc_4_4) (clear loc_4_5)
        (clear loc_5_1) (clear loc_5_2) (clear loc_5_3) (clear loc_5_4) (clear loc_5_5)

        ;; Adjacency predicates (horizontal and vertical for 5x5 grid)
        (adjacent loc_1_1 loc_1_2) (adjacent loc_1_2 loc_1_1)
        (adjacent loc_1_2 loc_1_3) (adjacent loc_1_3 loc_1_2)
        (adjacent loc_1_3 loc_1_4) (adjacent loc_1_4 loc_1_3)
        (adjacent loc_1_4 loc_1_5) (adjacent loc_1_5 loc_1_4)

        (adjacent loc_2_1 loc_2_2) (adjacent loc_2_2 loc_2_1)
        (adjacent loc_2_2 loc_2_3) (adjacent loc_2_3 loc_2_2)
        (adjacent loc_2_3 loc_2_4) (adjacent loc_2_4 loc_2_3)
        (adjacent loc_2_4 loc_2_5) (adjacent loc_2_5 loc_2_4)

        (adjacent loc_3_1 loc_3_2) (adjacent loc_3_2 loc_3_1)
        (adjacent loc_3_2 loc_3_3) (adjacent loc_3_3 loc_3_2)
        (adjacent loc_3_3 loc_3_4) (adjacent loc_3_4 loc_3_3)
        (adjacent loc_3_4 loc_3_5) (adjacent loc_3_5 loc_3_4)

        (adjacent loc_4_1 loc_4_2) (adjacent loc_4_2 loc_4_1)
        (adjacent loc_4_2 loc_4_3) (adjacent loc_4_3 loc_4_2)
        (adjacent loc_4_3 loc_4_4) (adjacent loc_4_4 loc_4_3)
        (adjacent loc_4_4 loc_4_5) (adjacent loc_4_5 loc_4_4)

        (adjacent loc_5_1 loc_5_2) (adjacent loc_5_2 loc_5_1)
        (adjacent loc_5_2 loc_5_3) (adjacent loc_5_3 loc_5_2)
        (adjacent loc_5_3 loc_5_4) (adjacent loc_5_4 loc_5_3)
        (adjacent loc_5_4 loc_5_5) (adjacent loc_5_5 loc_5_4)

        (adjacent loc_1_1 loc_2_1) (adjacent loc_2_1 loc_1_1)
        (adjacent loc_2_1 loc_3_1) (adjacent loc_3_1 loc_2_1)
        (adjacent loc_3_1 loc_4_1) (adjacent loc_4_1 loc_3_1)
        (adjacent loc_4_1 loc_5_1) (adjacent loc_5_1 loc_4_1)

        (adjacent loc_1_2 loc_2_2) (adjacent loc_2_2 loc_1_2)
        (adjacent loc_2_2 loc_3_2) (adjacent loc_3_2 loc_2_2)
        (adjacent loc_3_2 loc_4_2) (adjacent loc_4_2 loc_3_2)
        (adjacent loc_4_2 loc_5_2) (adjacent loc_5_2 loc_4_2)

        (adjacent loc_1_3 loc_2_3) (adjacent loc_2_3 loc_1_3)
        (adjacent loc_2_3 loc_3_3) (adjacent loc_3_3 loc_2_3)
        (adjacent loc_3_3 loc_4_3) (adjacent loc_4_3 loc_3_3)
        (adjacent loc_4_3 loc_5_3) (adjacent loc_5_3 loc_4_3)

        (adjacent loc_1_4 loc_2_4) (adjacent loc_2_4 loc_1_4)
        (adjacent loc_2_4 loc_3_4) (adjacent loc_3_4 loc_2_4)
        (adjacent loc_3_4 loc_4_4) (adjacent loc_4_4 loc_3_4)
        (adjacent loc_4_4 loc_5_4) (adjacent loc_5_4 loc_4_4)

        (adjacent loc_1_5 loc_2_5) (adjacent loc_2_5 loc_1_5)
        (adjacent loc_2_5 loc_3_5) (adjacent loc_3_5 loc_2_5)
        (adjacent loc_3_5 loc_4_5) (adjacent loc_4_5 loc_3_5)
        (adjacent loc_4_5 loc_5_5) (adjacent loc_5_5 loc_4_5)

        ;; Straight predicates (all horizontal and vertical 3-cell lines for 5x5 grid)
        ;; Horizontal
        (straight loc_1_1 loc_1_2 loc_1_3) (straight loc_1_3 loc_1_2 loc_1_1)
        (straight loc_1_2 loc_1_3 loc_1_4) (straight loc_1_4 loc_1_3 loc_1_2)
        (straight loc_1_3 loc_1_4 loc_1_5) (straight loc_1_5 loc_1_4 loc_1_3)

        (straight loc_2_1 loc_2_2 loc_2_3) (straight loc_2_3 loc_2_2 loc_2_1)
        (straight loc_2_2 loc_2_3 loc_2_4) (straight loc_2_4 loc_2_3 loc_2_2)
        (straight loc_2_3 loc_2_4 loc_2_5) (straight loc_2_5 loc_2_4 loc_2_3)

        (straight loc_3_1 loc_3_2 loc_3_3) (straight loc_3_3 loc_3_2 loc_3_1)
        (straight loc_3_2 loc_3_3 loc_3_4) (straight loc_3_4 loc_3_3 loc_3_2)
        (straight loc_3_3 loc_3_4 loc_3_5) (straight loc_3_5 loc_3_4 loc_3_3)

        (straight loc_4_1 loc_4_2 loc_4_3) (straight loc_4_3 loc_4_2 loc_4_1)
        (straight loc_4_2 loc_4_3 loc_4_4) (straight loc_4_4 loc_4_3 loc_4_2)
        (straight loc_4_3 loc_4_4 loc_4_5) (straight loc_4_5 loc_4_4 loc_4_3)

        (straight loc_5_1 loc_5_2 loc_5_3) (straight loc_5_3 loc_5_2 loc_5_1)
        (straight loc_5_2 loc_5_3 loc_5_4) (straight loc_5_4 loc_5_3 loc_5_2)
        (straight loc_5_3 loc_5_4 loc_5_5) (straight loc_5_5 loc_5_4 loc_5_3)

        ;; Vertical
        (straight loc_1_1 loc_2_1 loc_3_1) (straight loc_3_1 loc_2_1 loc_1_1)
        (straight loc_2_1 loc_3_1 loc_4_1) (straight loc_4_1 loc_3_1 loc_2_1)
        (straight loc_3_1 loc_4_1 loc_5_1) (straight loc_5_1 loc_4_1 loc_3_1)

        (straight loc_1_2 loc_2_2 loc_3_2) (straight loc_3_2 loc_2_2 loc_1_2)
        (straight loc_2_2 loc_3_2 loc_4_2) (straight loc_4_2 loc_3_2 loc_2_2)
        (straight loc_3_2 loc_4_2 loc_5_2) (straight loc_5_2 loc_4_2 loc_3_2)

        (straight loc_1_3 loc_2_3 loc_3_3) (straight loc_3_3 loc_2_3 loc_1_3)
        (straight loc_2_3 loc_3_3 loc_4_3) (straight loc_4_3 loc_3_3 loc_2_3)
        (straight loc_3_3 loc_4_3 loc_5_3) (straight loc_5_3 loc_4_3 loc_3_3)

        (straight loc_1_4 loc_2_4 loc_3_4) (straight loc_3_4 loc_2_4 loc_1_4)
        (straight loc_2_4 loc_3_4 loc_4_4) (straight loc_4_4 loc_3_4 loc_2_4)
        (straight loc_3_4 loc_4_4 loc_5_4) (straight loc_5_4 loc_4_4 loc_3_4)

        (straight loc_1_5 loc_2_5 loc_3_5) (straight loc_3_5 loc_2_5 loc_1_5)
        (straight loc_2_5 loc_3_5 loc_4_5) (straight loc_4_5 loc_3_5 loc_2_5)
        (straight loc_3_5 loc_4_5 loc_5_5) (straight loc_5_5 loc_4_5 loc_3_5)
    )

    (:goal (and
        (box-at box_0 loc_4_4)
        (box-at box_1 loc_4_5)
        (heavybox-at heavy_0 loc_1_5)
    ))
)