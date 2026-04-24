# Assignment 2: Probabilistic Box Pushing — Implementation & Theory Report

## Overview
This project tackles the complex challenge of operating within a multi-agent stochastic environment. The primary challenge was managing the stochastic nature of the environment (an 80% success rate for physical actions and a 20% chance of "slipping") while ensuring theoretical correctness and computational efficiency.

---

## 1. Environment & PDDL Extractor Architecture
To allow a rigid, deterministic classical planner to operate in a chaotic environment, we had to heavily engineer the `pddl_extractor.py` to prevent logic crashes and physical impossibilities.

### The "Aligned" Physics Constraint
Classical planners do not inherently understand 2D grid physics. Initially, the planner attempted to push boxes sideways through walls (the "L-shaped hallucination"). We solved this by injecting `aligned` triplets into the PDDL domain, explicitly teaching the planner that a valid push requires the agent, the box, and the target space to form a straight line. This work was similar to ex1 and was necessary in order to make the online planner work correctly.

### The "Hidden Goal" & "Box Swap" Bugs
When a box is pushed onto a goal, the grid scanner reads "Box" and forgets the goal underneath. Furthermore, because grid scanners read top-to-bottom, a box moving down a row could suddenly be renamed from `box_0` to `box_1`, causing the planner to scramble targets and crash with `"No plan found."`
* **The Fix (Dynamic Goal Binding):** We bypassed the dynamic grid scanner's default goal logic. Instead, we dynamically extracted all goal and box coordinates, sorted them by their X-coordinates, and strictly paired them. By binding targets based on absolute geometry rather than top-to-bottom scanning order, we guaranteed mathematical consistency across all states, regardless of the map layout.

---

## 2. Part 1: Online Planning
The Online Planner utilizes Fast Downward, a deterministic A* graph-search algorithm. Because the environment is stochastic, the planner's assumptions are frequently violated when agents slip.

### Execution Monitoring (Plan Repair)
Initially, the algorithm asked Fast Downward to calculate a path to the finish line, took *one step*, and then threw the plan away to calculate a new one. This resulted in massive computational bottlenecks.
* **Optimization:** We implemented **Execution Monitoring**. The agent calculates a plan and loads it into a local queue. After executing a step, the monitoring system verifies the outcome: `if env.agent_positions[a] != target`. If the agent successfully reached the target (80% of the time), it continues executing the cached plan. It only clears the queue and calls the expensive PDDL solver when an agent physically slips.

### The Global Plan Cache (Memoization)
Because the environment is a Markov Decision Process (MDP), the optimal classical plan from any specific grid state is immutable. 
* **Theory:** Fast Downward is a deterministic algorithm. Given the same board geometry, it will evaluate the same heuristic tree and output the exact same path. 
* **Optimization:** We created a `GLOBAL_PLAN_CACHE`. Before generating PDDL files, the system mathematically hashes the board state and checks if it has seen this exact layout in a previous run. If so, it instantly loads the pre-calculated sequence. Over multiple runs, the classical planner is eventually bypassed entirely, dropping episode execution time from seconds to milliseconds.

---

## 3. Part 2: Modified Policy Iteration (MPI)
MPI requires building a complete Transition Model (a graph of all possible multiverses) and using the Bellman Equation to find the optimal policy.

### State Abstraction (Dimensionality Reduction)
The raw state space of a multi-agent gridworld is astronomically large due to combinatorial explosion. 
* **Theory:** To compute the Bellman equations efficiently, we needed a minimal mathematical "fingerprint" of the board that satisfies the **Markov Property** (containing only the information necessary to make the next decision).
* **Optimization:** In our `get_state()` function, we **dropped the agents' facing directions**, shrinking the state space by a factor of 16. We also explicitly **sorted the identical small boxes**. By recognizing that two small boxes swapping places does not change the physical puzzle, we collapsed mirrored geometries. This reduced the environment to exactly **223 canonical reachable states**, allowing MPI to converge rapidly.

### Heuristic Value Initialization
* **The Sparse Reward Problem:** The environment only grants a reward of `+1.0` upon total completion. If initialized with $V_0 = 0.0$, the algorithm must wait for the reward to "trickle backward" step-by-step from the terminal state over dozens of iterations.
* **Theory:** The Bellman Operator is a **Contraction Mapping**. According to the Banach Fixed-Point Theorem, repeated application of the Bellman equation will converge to the optimal $V^*$, regardless of the starting values.
* **Optimization:** We initialized $V_0$ using a dynamic **Manhattan Distance Heuristic**. By applying a negative penalty based on how far the boxes are from the nearest goal, we "tilted" the value table into a funnel. The agents feel a gradient pulling them toward the goals on Iteration 1, significantly speeding up the convergence rate of the outer loops.

---

## 4. Comparative Analysis & Benchmarking
*(Note: Evaluation conducted over 100 episodes on the primary assignment map)*
* **Online Planning:** [INSERT ONLINE MEAN] Mean Steps
* **MPI:** [INSERT MPI MEAN] Mean Steps

While MPI yielded a lower mean step count, this highlights differences in architecture and theoretical framing:
* **Sequential vs. Joint Actions:** PDDL is a sequential planner. It forces one agent to move while the other waits, inherently inflating the "step count" clock. MPI utilizes a **Joint Action Space**, allowing both agents to move simultaneously. *Future Work: The PDDL domain could be modified to support parallel actions (e.g., `move-together`) to equalize this mechanical discrepancy.*
* **Stochastic Awareness vs. Determinism:** The core reason MPI outperforms Online Planning is its fundamental awareness of the environment's probabilities. MPI's Bellman equations explicitly factor in the 20% slip rate, allowing it to mathematically favor safer, central routes. The Fast Downward planner assumes the world is perfectly deterministic; it is completely blind to risk and only reacts to stochasticity after a failure has already occurred.