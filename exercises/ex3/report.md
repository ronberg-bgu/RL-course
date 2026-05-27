# Programming Assignment 3: RL Algorithms Comparison

## 1. Project Introduction
This project compares three Reinforcement Learning algorithms.
We evaluate Tabular Q-learning, Tabular SARSA, and REINFORCE.
The environment is Gymnasium's `FrozenLake-v1` on an 8x8 grid.
The environment is stochastic (`is_slippery=True`).
This adds severe noise, as intended actions succeed only 33% of the time.

### Included Files
* `algorithms.py`: Contains the core algorithm implementations.
  It also handles the evaluation loop and graph generation.
* `.env`: A configuration file holding all hyperparameters.
  This allows easy tuning without modifying the Python code.

---

## 2. Hyperparameters and Tuning

### General Parameters
* `ENV_NAME=FrozenLake-v1`: The selected environment.
* `IS_SLIPPERY=True`: Enables stochastic transitions.
* `MAX_STEPS_PER_EPISODE=200`: Prevents infinite loops.
* `TOTAL_TRAINING_STEPS=500000`: Total environment steps for training.
* `EVAL_INTERVAL_STEPS=10000`: Steps between each evaluation phase.
* `N_EVAL_EPISODES=500`: Episodes run to estimate the current value.
* `GAMMA=0.99`: The discount factor for future rewards.

### Exploration Parameters
* `EPSILON_START=1.0`
* `EPSILON_END=0.05`
* `EPSILON_DECAY_STEPS=300000`
  *Note:* A long decay is crucial here. The reward is extremely sparse.
  If exploration ends too early, the agent converges to a 0.0 return.

### Tuned Parameters and Justification
* **`ALPHA=0.05`** (Changed from baseline 0.1)
  * *Theoretical:* A high learning rate in a noisy environment is bad.
    It makes Q-values overly sensitive to random slips.
  * *Empirical:* Lowering alpha smoothed out the updates.
    It prevented Q-learning from crashing after finding a good path.
* **`POLICY_LR=0.01`** (Changed from baseline 0.005)
  * *Theoretical:* REINFORCE without a baseline suffers from huge variance.
    A learning rate of 0.005 is too conservative for sparse rewards.
  * *Empirical:* Increasing it allowed the policy to properly update
    when the agent occasionally reached the goal.
* **`THETA_CLIP=20.0`**
  * Prevents numerical overflow in the softmax function.

---

## 3. Evaluation of V(s0)

The empirical value of the initial state, V(s0), is calculated as follows:
1. Training is paused every 10,000 steps.
2. A separate evaluation environment is initialized.
3. The current policy is frozen (no updates occur).
4. The agent runs 500 episodes starting from the initial state.
5. Action selection is purely greedy for all algorithms.
6. The discounted return is calculated for each episode.
7. V(s0) is the arithmetic mean of these 500 returns.

---

## 4. Discussion of Results

Based on the generated graphs across multiple random seeds:

* **Learning Speed:** Q-learning initially learns the fastest. 
  It shows the sharpest rise in performance early in training.

* **Best Performance & Stability:** SARSA achieved the best overall stability and final performance.
  It converges smoothly and maintains its learned value.

* **Q-learning vs. SARSA:** Q-learning is off-policy and assumes optimal future actions.
  It learns an optimistic, risky path near the holes.
  When it randomly slips, it receives severe penalties and crashes.
  SARSA is on-policy and factors in the random exploration.
  It learns a pessimistic, safer, and longer path.
  This makes SARSA much more robust to the slippery ice.

* **REINFORCE Stability:** REINFORCE was the slowest and least stable algorithm.
  Because it lacks a value baseline, the sparse rewards cause issues.
  The Monte Carlo updates suffer from massive variance.

* **Impact of Stochasticity:** The slippery ice makes the environment highly unpredictable.
  It forces agents to value safety over the shortest path.
  It also drastically reduces the maximum possible return, 
  as even a perfect policy will often slip and fail.