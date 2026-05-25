# Exercise 3 — Comparison of Tabular RL Algorithms on FrozenLake-v1

## Submitted by:
Pinhas Aburmad 212146849  
Segev Olpak 325176188

## Overview

This exercise implements and compares three tabular Reinforcement Learning algorithms on the **FrozenLake-v1** environment (8×8 grid, `is_slippery=True`):

1. **Tabular Q-learning** — Off-policy temporal-difference control  
2. **Tabular SARSA** — On-policy temporal-difference control  
3. **REINFORCE** — Monte-Carlo policy gradient with tabular softmax parameterisation  

All algorithms are trained for **500,000 environment steps** across **5 random seeds** and periodically evaluated to estimate the value of the initial state, $V(s_0)$.

---

## How to Run

```bash
cd RL-course
python exercises/ex3/solution_ex3.py
```

The script will train all three algorithms, print a summary of final performance, and save the comparison plot to `exercises/ex3/comparison_plot.png`.

---

## How $V(s_0)$ Was Evaluated

Every **10,000 environment steps**, training is paused and the current policy is frozen for evaluation.  The agent runs **500 evaluation episodes** starting from state $s_0$ (the top-left corner of the grid).  During each episode:

1. The agent follows a **completely greedy** policy (no exploration noise):
   - For Q-learning and SARSA: `action = argmax_a Q(s, a)`
   - For REINFORCE: `action = argmax_a π(a | s)` where π is the softmax policy
2. The **discounted return** $G = \sum_{t=0}^{T} \gamma^t r_t$ is computed for the full episode.
3. The mean of these 500 returns gives $\hat{V}(s_0)$.

Evaluation steps do **not** count towards the training step budget.

---

## Discussion of Results

### Which algorithm learned faster and which achieved better final performance?

**Q-learning** typically learns the fastest among the three algorithms and achieves the highest final $V(s_0)$.  This is expected because Q-learning's off-policy nature allows it to learn from the maximum future Q-value regardless of the action actually taken, making it more sample-efficient in environments where exploration is separate from exploitation.

**SARSA** follows closely but generally converges to a slightly lower value.  Its on-policy nature means it learns values consistent with the exploratory behaviour policy (which includes ε-greedy randomness), leading to somewhat more conservative value estimates during training.  However, when evaluated with a greedy policy, its final performance is often comparable to Q-learning.

**REINFORCE** is the slowest to learn and typically achieves lower final performance.  As a Monte-Carlo method, it must wait until the end of each episode to update, making it significantly less sample-efficient.

### Did Q-learning and SARSA behave differently?

Yes, though the differences are nuanced in FrozenLake:

- **Q-learning** is optimistic — it bootstraps from `max_a Q(s', a)`, learning about the optimal policy directly.  This leads to faster convergence toward the true optimal value.
- **SARSA** is more conservative — it bootstraps from the actual next action taken under the behaviour policy, which includes exploration.  During early training (high ε), SARSA's Q-values are depressed because they incorporate the cost of exploratory actions.

In practice on FrozenLake, both algorithms eventually converge to similar performance once ε decays to its minimum value (0.05), but Q-learning's learning curve rises earlier.

### Was REINFORCE less stable?

**Yes, significantly.** REINFORCE exhibits much higher variance across seeds, visible as wider shaded regions in the plot.  Several factors contribute:

1. **High-variance gradient estimates:** REINFORCE uses full episode returns to compute policy gradients.  Without a baseline, these gradients have high variance, especially early in training.
2. **No bootstrapping:** Unlike TD methods, REINFORCE cannot learn from partial episodes.  In FrozenLake, episodes can be very long (up to 200 steps) with only a terminal reward, so the signal-to-noise ratio is very low.
3. **Sparse reward:** The only non-zero reward is +1 for reaching the goal.  Early on, the agent rarely reaches the goal, so most gradient updates carry no useful reward signal.

### How did the stochastic nature of FrozenLake impact learning?

The `is_slippery=True` setting means each action has only a 1/3 probability of executing as intended; the agent slips to a perpendicular direction with the remaining 2/3 probability.  This has several consequences:

1. **Lower optimal V(s₀):** Even a perfect policy achieves a relatively low success rate (the theoretical optimal $V(s_0)$ on slippery 8×8 FrozenLake is roughly 0.7–0.85), because the agent inevitably slips into holes some fraction of the time.
2. **Slower convergence:** The stochasticity creates a much noisier signal — the same (state, action) pair leads to different next states, so the Q-values and policy gradients require many more samples to converge.
3. **Harder credit assignment:** The combination of sparse rewards and stochastic transitions makes it difficult for all algorithms to determine which actions were actually helpful.
4. **REINFORCE suffers most:** Policy-gradient methods are particularly sensitive to stochasticity because they rely on full-trajectory returns.  The added transition noise amplifies the already-high variance of Monte-Carlo gradient estimates.

---

## Hyperparameter Summary

| Parameter | Value |
|---|---|
| γ (discount factor) | 0.99 |
| max_steps_per_episode | 200 |
| total_training_steps | 500,000 |
| seeds | [0, 1, 2, 3, 4] |
| α (Q-learning/SARSA) | 0.1 |
| ε_start | 1.0 |
| ε_end | 0.05 |
| ε_decay_steps | 300,000 |
| policy_lr (REINFORCE) | 0.005 |
| θ_init | 0.0 |
| θ_clip | 20.0 |
| eval_interval | 10,000 steps |
| eval_episodes | 500 |
