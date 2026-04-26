# Assignment 2: Probabilistic Box Pushing — Implementation & Theory Report

## Overview
This project tackles the complex challenge of operating within a multi-agent stochastic environment. The primary challenge was managing the stochastic nature of the environment (an 80% success rate for physical actions and a 20% chance of "slipping") while ensuring theoretical correctness and computational efficiency.

## How to Run
To execute the evaluation and generate the performance comparison, ensure you are in the `ex2` directory and run the following command in your terminal:

```bash
python .\exercises\ex2\solution_ex2.py
```

*Note: The first few runs of the Online Planner may be slightly slower as the `GLOBAL_PLAN_CACHE` is populated. The MPI algorithm will then trigger its BFS state-space mapping before beginning the policy iteration convergence.*

---

## 1. Environment & PDDL Extractor Architecture
To allow a rigid, deterministic classical planner to operate in a chaotic environment, we had to heavily engineer the `pddl_extractor.py` and adjust the base environment to prevent logic crashes and physical impossibilities.

### The "Aligned" Physics Constraint
Classical planners do not inherently understand 2D grid physics. Initially, the planner attempted to push boxes sideways through walls (the "L-shaped hallucination"). We solved this by injecting `aligned` triplets into the PDDL domain, explicitly teaching the planner that a valid push requires the agent, the box, and the target space to form a straight line. This work was similar to Ex1 and was necessary in order to make the online planner work correctly.

### Refining the Termination Condition
During our initial testing, we noticed a slight quirk in the base environment's logic where the episode would prematurely terminate as soon as a single box was pushed onto a goal. To ensure the algorithms were evaluated on the complete multi-agent task, we gently adjusted the environment's termination condition. It now accurately checks that *all* boxes have successfully reached their respective goals before concluding the episode, allowing for a proper evaluation of the full plan.

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

### Reachability Graph Search (BFS)
Rather than using a naive nested `for` loop to iterate over every possible permutation of grid coordinates—which would generate a massive number of physically impossible configurations—we implemented a targeted Breadth-First Search (BFS). Starting strictly from the initial state, the BFS simulates valid actions to dynamically discover and map only the *physically reachable* states. This drastically pruned the state space before the Bellman updates even began.

### State Abstraction (Dimensionality Reduction)
Even within physically reachable bounds, the raw state space of a multi-agent gridworld is incredibly large. 
* **Theory:** To compute the Bellman equations efficiently, we needed a minimal mathematical "fingerprint" of the board that satisfies the **Markov Property** (containing only the information necessary to make the next decision).
* **Optimization:** In our `get_state()` function, we **dropped the agents' facing directions**, shrinking the state space by a factor of 16. We also explicitly **sorted the identical small boxes**. By recognizing that two small boxes swapping places does not change the physical puzzle, we significantly reduced the state space.

### Combating the Curse of Dimensionality (Integer State Mapping)
* **The Memory Bottleneck:** Even with rigorous state abstraction, multi-agent grid environments inherently suffer from the curse of dimensionality. Furthermore, in Python, complex nested tuples (e.g., storing the distinct coordinates of multiple agents and boxes) carry massive object overhead. Storing hundreds of thousands of these heavy tuples inside the Transition Model rapidly exhausts system RAM.
* **Theory & Optimization:** To resolve this, we implemented an **Integer State Mapping** architecture. During the BFS, every newly discovered state tuple is mathematically hashed into a single, primitive integer ID. The massive `transitions` graph and the Value table (`V`) are built entirely using these lightweight integer IDs rather than the heavy objects. 
* **Result:** This creates a single source of truth for the bulky state tuples, shrinking the algorithm's memory footprint by orders of magnitude. As a secondary benefit, integer-based dictionary lookups execute significantly faster than recursive nested-tuple comparisons, vastly accelerating the Bellman update loops. Once convergence is achieved, the integer keys are seamlessly unpacked back into their physical coordinate tuples for the final policy evaluation.

### Heuristic Value Initialization
* **The Sparse Reward Problem:** The environment only grants a reward of `+1.0` upon total completion. If initialized with $V_0 = 0.0$, the algorithm must wait for the reward to "trickle backward" step-by-step from the terminal state over dozens of iterations.
* **Theory:** The Bellman Operator is a **Contraction Mapping**. According to the Banach Fixed-Point Theorem, repeated application of the Bellman equation will converge to the optimal $V^*$, regardless of the starting values.
* **Optimization:** We initialized $V_0$ by applying a dynamic penalty based on the total grid distance (horizontal plus vertical steps) between the boxes and their nearest goals. By applying this distance as a negative value, we effectively "tilted" the value table into a funnel. The agents feel a gradient pulling them toward the goals on Iteration 1, significantly speeding up the convergence rate of the outer loops.

---

## 4. Comparative Analysis & Benchmarking
*(Note: Evaluation conducted over 100 episodes on a single map)*
* **Online Planning:** 40.62 Mean Steps (Std: 8.88)
* **MPI:** 24.59 Mean Steps (Std: 4.20)

While MPI yielded a lower mean step count, this highlights differences in architecture and theoretical framing:
* **Sequential vs. Joint Actions:** PDDL is a sequential planner. It forces one agent to move while the other waits, inherently inflating the "step count" clock. MPI utilizes a **Joint Action Space**, allowing both agents to move simultaneously. *(Note on future expansions: If a strictly 1:1 mechanical comparison is ever required, extra future work could involve extending the PDDL domain to support parallel actions, thereby equalizing this discrepancy).*
* **Stochastic Awareness vs. Determinism:** The core reason MPI outperforms Online Planning is its fundamental awareness of the environment's probabilities. MPI's Bellman equations explicitly factor in the 20% slip rate, allowing it to mathematically favor safer, central routes. The Fast Downward planner assumes the world is perfectly deterministic; it is completely blind to risk and only reacts to stochasticity after a failure has already occurred.
* **Variance and Risk Assessment (Standard Deviation):** The standard deviation across runs reveals how each algorithm interacts with map topology. While it is tempting to assume MPI's complete Transition Model yields lower variance, our empirical results show standard deviation fluctuates depending on the map. MPI optimizes strictly for the lowest *expected* step count; therefore, it will willingly execute "risky" maneuvers (e.g., moving near obstacles where slipping causes severe delays) if the mathematical expected value outperforms a longer, safer detour. Conversely, the Online Planner assumes a deterministic world. Its variance is dictated less by calculated risk and more by how aggressively a stochastic slip breaks its "perfect world" sequence, forcing it into localized replanning loops.
