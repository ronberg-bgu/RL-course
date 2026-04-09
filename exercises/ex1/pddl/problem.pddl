(define (problem box-pushing-4x4)
  (:domain box-pushing)

  (:objects
    agent_0 agent_1 - agent
    loc_1_1 loc_1_2 loc_1_3 loc_1_4
    loc_2_1 loc_2_2 loc_2_3 loc_2_4
    loc_3_1 loc_3_2 loc_3_3 loc_3_4
    loc_4_1 loc_4_2 loc_4_3 loc_4_4 - location
    box_0 box_1 - box
    hbx_0 - heavybox
  )

  (:init
    ;; Adjacency
    (adj loc_1_1 loc_1_2) (adj loc_1_2 loc_1_1)
    (adj loc_1_2 loc_1_3) (adj loc_1_3 loc_1_2)
    (adj loc_1_3 loc_1_4) (adj loc_1_4 loc_1_3)

    (adj loc_2_1 loc_2_2) (adj loc_2_2 loc_2_1)
    (adj loc_2_2 loc_2_3) (adj loc_2_3 loc_2_2)
    (adj loc_2_3 loc_2_4) (adj loc_2_4 loc_2_3)

    (adj loc_3_1 loc_3_2) (adj loc_3_2 loc_3_1)
    (adj loc_3_2 loc_3_3) (adj loc_3_3 loc_3_2)
    (adj loc_3_3 loc_3_4) (adj loc_3_4 loc_3_3)

    (adj loc_4_1 loc_4_2) (adj loc_4_2 loc_4_1)
    (adj loc_4_2 loc_4_3) (adj loc_4_3 loc_4_2)
    (adj loc_4_3 loc_4_4) (adj loc_4_4 loc_4_3)

    (adj loc_1_1 loc_2_1) (adj loc_2_1 loc_1_1)
    (adj loc_2_1 loc_3_1) (adj loc_3_1 loc_2_1)
    (adj loc_3_1 loc_4_1) (adj loc_4_1 loc_3_1)

    (adj loc_1_2 loc_2_2) (adj loc_2_2 loc_1_2)
    (adj loc_2_2 loc_3_2) (adj loc_3_2 loc_2_2)
    (adj loc_3_2 loc_4_2) (adj loc_4_2 loc_3_2)

    (adj loc_1_3 loc_2_3) (adj loc_2_3 loc_1_3)
    (adj loc_2_3 loc_3_3) (adj loc_3_3 loc_2_3)
    (adj loc_3_3 loc_4_3) (adj loc_4_3 loc_3_3)

    (adj loc_1_4 loc_2_4) (adj loc_2_4 loc_1_4)
    (adj loc_2_4 loc_3_4) (adj loc_3_4 loc_2_4)
    (adj loc_3_4 loc_4_4) (adj loc_4_4 loc_3_4)

    ;; Straight push triples
    (in-line loc_1_1 loc_2_1 loc_3_1) (in-line loc_3_1 loc_2_1 loc_1_1)
    (in-line loc_2_1 loc_3_1 loc_4_1) (in-line loc_4_1 loc_3_1 loc_2_1)
    (in-line loc_1_2 loc_2_2 loc_3_2) (in-line loc_3_2 loc_2_2 loc_1_2)
    (in-line loc_2_2 loc_3_2 loc_4_2) (in-line loc_4_2 loc_3_2 loc_2_2)
    (in-line loc_1_3 loc_2_3 loc_3_3) (in-line loc_3_3 loc_2_3 loc_1_3)
    (in-line loc_2_3 loc_3_3 loc_4_3) (in-line loc_4_3 loc_3_3 loc_2_3)
    (in-line loc_1_4 loc_2_4 loc_3_4) (in-line loc_3_4 loc_2_4 loc_1_4)
    (in-line loc_2_4 loc_3_4 loc_4_4) (in-line loc_4_4 loc_3_4 loc_2_4)
    (in-line loc_1_1 loc_1_2 loc_1_3) (in-line loc_1_3 loc_1_2 loc_1_1)
    (in-line loc_1_2 loc_1_3 loc_1_4) (in-line loc_1_4 loc_1_3 loc_1_2)
    (in-line loc_2_1 loc_2_2 loc_2_3) (in-line loc_2_3 loc_2_2 loc_2_1)
    (in-line loc_2_2 loc_2_3 loc_2_4) (in-line loc_2_4 loc_2_3 loc_2_2)
    (in-line loc_3_1 loc_3_2 loc_3_3) (in-line loc_3_3 loc_3_2 loc_3_1)
    (in-line loc_3_2 loc_3_3 loc_3_4) (in-line loc_3_4 loc_3_3 loc_3_2)
    (in-line loc_4_1 loc_4_2 loc_4_3) (in-line loc_4_3 loc_4_2 loc_4_1)
    (in-line loc_4_2 loc_4_3 loc_4_4) (in-line loc_4_4 loc_4_3 loc_4_2)

    ;; Clear locations (all except box locations)
    (clear loc_1_1)
    (clear loc_1_2)
    (clear loc_1_4)
    (clear loc_2_1)
    (clear loc_2_2)
    (clear loc_2_4)
    (clear loc_3_1)
    (clear loc_3_2)
    (clear loc_3_3)
    (clear loc_3_4)
    (clear loc_4_1)
    (clear loc_4_2)
    (clear loc_4_4)

    ;; Agent positions
    (agent-at agent_0 loc_1_1)
    (agent-at agent_1 loc_2_1)

    ;; Box positions
    (box-at box_0 loc_1_3)
    (box-at box_1 loc_4_3)
    (heavybox-at hbx_0 loc_2_3)
  )

  (:goal
    (and
      (box-at box_0 loc_1_4)
      (box-at box_1 loc_4_4)
      (heavybox-at hbx_0 loc_2_4))
  )
)