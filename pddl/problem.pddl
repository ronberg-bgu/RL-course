(define (problem box-pushing-5x5)
    (:domain box-pushing)
    
    (:objects
        agent_0 agent_1 - agent
        box_0 box_1 - box
        hbx_0 - heavybox
        loc_1_1 loc_1_2 loc_1_3 loc_1_4 loc_1_5 - location
        loc_2_1 loc_2_2 loc_2_3 loc_2_4 loc_2_5 - location
        loc_3_1 loc_3_2 loc_3_3 loc_3_4 loc_3_5 - location
        loc_4_1 loc_4_2 loc_4_3 loc_4_4 loc_4_5 - location
        loc_5_1 loc_5_2 loc_5_3 loc_5_4 loc_5_5 - location
    )
    
    (:init
        ;; 1. Initial Entity Placements
        (at-agent agent_0 loc_1_1)
        (at-agent agent_1 loc_1_1)
        
        (at-box box_0 loc_2_3)
        (at-box box_1 loc_3_2)
        (at-heavybox hbx_0 loc_4_2)

        ;; 2. Clear Designations
        ;; All grid tiles except obstacles (loc_2_2, loc_4_4) and initial box locations
        (clear loc_1_1) (clear loc_1_2) (clear loc_1_3) (clear loc_1_4) (clear loc_1_5)
        (clear loc_2_1)                 (clear loc_2_4) (clear loc_2_5)
        (clear loc_3_1)                 (clear loc_3_3) (clear loc_3_4) (clear loc_3_5)
        (clear loc_4_1)                 (clear loc_4_3)                 (clear loc_4_5)
        (clear loc_5_1) (clear loc_5_2) (clear loc_5_3) (clear loc_5_4) (clear loc_5_5)

        ;; 3. Adjacency (Bidirectional) - Obstacles loc_2_2 and loc_4_4 are omitted
        ;; Horizontal
        (adj loc_1_1 loc_1_2) (adj loc_1_2 loc_1_1) (adj loc_1_2 loc_1_3) (adj loc_1_3 loc_1_2)
        (adj loc_1_3 loc_1_4) (adj loc_1_4 loc_1_3) (adj loc_1_4 loc_1_5) (adj loc_1_5 loc_1_4)
        
        (adj loc_2_3 loc_2_4) (adj loc_2_4 loc_2_3) (adj loc_2_4 loc_2_5) (adj loc_2_5 loc_2_4)
        
        (adj loc_3_1 loc_3_2) (adj loc_3_2 loc_3_1) (adj loc_3_2 loc_3_3) (adj loc_3_3 loc_3_2)
        (adj loc_3_3 loc_3_4) (adj loc_3_4 loc_3_3) (adj loc_3_4 loc_3_5) (adj loc_3_5 loc_3_4)
        
        (adj loc_4_1 loc_4_2) (adj loc_4_2 loc_4_1) (adj loc_4_2 loc_4_3) (adj loc_4_3 loc_4_2)
        
        (adj loc_5_1 loc_5_2) (adj loc_5_2 loc_5_1) (adj loc_5_2 loc_5_3) (adj loc_5_3 loc_5_2)
        (adj loc_5_3 loc_5_4) (adj loc_5_4 loc_5_3) (adj loc_5_4 loc_5_5) (adj loc_5_5 loc_5_4)

        ;; Vertical
        (adj loc_1_1 loc_2_1) (adj loc_2_1 loc_1_1) (adj loc_2_1 loc_3_1) (adj loc_3_1 loc_2_1)
        (adj loc_3_1 loc_4_1) (adj loc_4_1 loc_3_1) (adj loc_4_1 loc_5_1) (adj loc_5_1 loc_4_1)
        
        (adj loc_3_2 loc_4_2) (adj loc_4_2 loc_3_2) (adj loc_4_2 loc_5_2) (adj loc_5_2 loc_4_2)
        
        (adj loc_1_3 loc_2_3) (adj loc_2_3 loc_1_3) (adj loc_2_3 loc_3_3) (adj loc_3_3 loc_2_3)
        (adj loc_3_3 loc_4_3) (adj loc_4_3 loc_3_3) (adj loc_4_3 loc_5_3) (adj loc_5_3 loc_4_3)
        
        (adj loc_1_4 loc_2_4) (adj loc_2_4 loc_1_4) (adj loc_2_4 loc_3_4) (adj loc_3_4 loc_2_4)
        
        (adj loc_1_5 loc_2_5) (adj loc_2_5 loc_1_5) (adj loc_2_5 loc_3_5) (adj loc_3_5 loc_2_5)
        (adj loc_3_5 loc_4_5) (adj loc_4_5 loc_3_5) (adj loc_4_5 loc_5_5) (adj loc_5_5 loc_4_5)

        ;; 4. In-Line Relationships (Bidirectional Straight Paths) - Omit if passing an obstacle
        ;; Horizontal Lines
        (in-line loc_1_1 loc_1_2 loc_1_3) (in-line loc_1_3 loc_1_2 loc_1_1)
        (in-line loc_1_2 loc_1_3 loc_1_4) (in-line loc_1_4 loc_1_3 loc_1_2)
        (in-line loc_1_3 loc_1_4 loc_1_5) (in-line loc_1_5 loc_1_4 loc_1_3)
        
        (in-line loc_2_3 loc_2_4 loc_2_5) (in-line loc_2_5 loc_2_4 loc_2_3)
        
        (in-line loc_3_1 loc_3_2 loc_3_3) (in-line loc_3_3 loc_3_2 loc_3_1)
        (in-line loc_3_2 loc_3_3 loc_3_4) (in-line loc_3_4 loc_3_3 loc_3_2)
        (in-line loc_3_3 loc_3_4 loc_3_5) (in-line loc_3_5 loc_3_4 loc_3_3)
        
        (in-line loc_4_1 loc_4_2 loc_4_3) (in-line loc_4_3 loc_4_2 loc_4_1)
        
        (in-line loc_5_1 loc_5_2 loc_5_3) (in-line loc_5_3 loc_5_2 loc_5_1)
        (in-line loc_5_2 loc_5_3 loc_5_4) (in-line loc_5_4 loc_5_3 loc_5_2)
        (in-line loc_5_3 loc_5_4 loc_5_5) (in-line loc_5_5 loc_5_4 loc_5_3)

        ;; Vertical Lines
        (in-line loc_1_1 loc_2_1 loc_3_1) (in-line loc_3_1 loc_2_1 loc_1_1)
        (in-line loc_2_1 loc_3_1 loc_4_1) (in-line loc_4_1 loc_3_1 loc_2_1)
        (in-line loc_3_1 loc_4_1 loc_5_1) (in-line loc_5_1 loc_4_1 loc_3_1)
        
        (in-line loc_3_2 loc_4_2 loc_5_2) (in-line loc_5_2 loc_4_2 loc_3_2)
        
        (in-line loc_1_3 loc_2_3 loc_3_3) (in-line loc_3_3 loc_2_3 loc_1_3)
        (in-line loc_2_3 loc_3_3 loc_4_3) (in-line loc_4_3 loc_3_3 loc_2_3)
        (in-line loc_3_3 loc_4_3 loc_5_3) (in-line loc_5_3 loc_4_3 loc_3_3)
        
        (in-line loc_1_4 loc_2_4 loc_3_4) (in-line loc_3_4 loc_2_4 loc_1_4)
        
        (in-line loc_1_5 loc_2_5 loc_3_5) (in-line loc_3_5 loc_2_5 loc_1_5)
        (in-line loc_2_5 loc_3_5 loc_4_5) (in-line loc_4_5 loc_3_5 loc_2_5)
        (in-line loc_3_5 loc_4_5 loc_5_5) (in-line loc_5_5 loc_4_5 loc_3_5)
    )

    (:goal (and
        (at-box box_0 loc_5_5)
        (at-box box_1 loc_5_4)
        (at-heavybox hbx_0 loc_3_3)
    ))
)