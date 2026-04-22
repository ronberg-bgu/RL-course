#we have the ppdl files in place, and the prompt here
prompt = """*Role:* You are an expert in Automated Planning and PDDL (Planning Domain Definition Language).
*Task:* Create a domain.pddl and a problem.pddl for a grid-based logistics simulation called "Box Pushing."
### 1. The World Blueprint
Define the following hierarchy and naming conventions:
 * *Types:* Create a hierarchy where box and heavybox are distinct, alongside agent and location.
 * *Identity:* Use agent_0 and agent_1. Boxes are box_0 and box_1, while the heavy box is hbx_0.
 * *Coordinates:* Locations must follow the loc_x_y format (e.g., loc_1_1).
### 2. Physical Rules of the Environment
Please model the domain logic based on these natural-language behaviors:
 * *Shared Space:* The grid allows multiple agents to stand on the exact same tile at the same time. However, boxes and obstacles block tiles. A tile is only "clear" if no box (regular or heavy) is on it.
 * *Simple Movement:* An agent can walk to an adjacent tile if it is clear.
 * *Pushing Standard Boxes:* A single agent can push a regular box. To do this, the agent must be next to the box, and there must be an empty space on the opposite side of the box. As the agent pushes, they move into the spot where the box was, and the box slides into the empty space.
 * *The Heavy Box Challenge:* A heavybox is too heavy for one person. To move it, *both agents* must be standing on the same tile adjacent to the heavy box. They must then push together into an empty space. After the push, both agents occupy the box's previous tile, and the box occupies the destination.
### 3. Required Action Signatures
The actions in your domain.pddl *must* match these exact signatures:
 * *Move:* (:action move :parameters (?a - agent ?from - location ?to - location))
   * Logic: Agent moves from from to to. Requires adj.
 * *Push Small:* (:action push-small :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box))
   * Logic: Agent at from, box at boxloc, empty space at toloc.
 * *Push Heavy:* (:action push-heavy :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox))
   * Logic: Two different agents at from, heavy box at boxloc, empty space at toloc.
### 4. The Scenario (Problem File)
Generate a problem.pddl for a *5x5 grid* with the following setup:
 * *Initialization:* * Define the full adjacency map (up, down, left, right).
   * Place *two obstacles* by simply omitting their clear status and adjacency links.
   * Start agent_0 and agent_1 at loc_1_1.
   * Scatter box_0, box_1, and hbx_0 across the mid-section of the grid.
 * *The Goal:* The task is complete once the boxes reach these specific coordinates (the final agent positions do not matter):
   * box_0 at loc_5_5
   * box_1 at loc_5_4
   * hbx_0 at loc_3_3
*Output Format:* Provide the complete code for domain.pddl and problem.pddl in two separate, clean code blocks. Ensure all clear predicates are correctly updated so tiles are freed or occupied accurately after every movement."""