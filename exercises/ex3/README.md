# Exercise 3

## V(s₀) Evaluation

Every **10,000 environment steps** the training session is paused, and the current policy is frozen for evaluation. Afterwards, the agent will run **500 evaluation episodes** from s0. 

1. The agent follows a completely greedy policy :
 Q-learning / SARSA: `action = argmax_a Q(s, a)`
 REINFORCE: `action = argmax_a softmax(theta[s])`
2. Returns `G = Σ (γ^t) r_t`.
3. Return the mean of 500 results for V(s0).


---

## Hyperparameters

| Parameter | Value |
|---|---|
| γ | 0.99 |
| α | 0.1 |
| ε_start | 1.0 |
| ε_end | 0.05 |
| ε_decay_steps | 300,000 |
| policy_lr | 0.005 |
| use_baseline | False |
| theta_init | 0.0 |
| theta_clip | 20.0 |
| total_training_steps | 500,000 |
| eval_interval | 10,000 steps |
| eval_episodes | 500 |
| max_steps_per_episode | 200 |
| seeds | [0, 1, 2, 3, 4] |

---

## Results

### Who learned faster and who achieved better final performance?

Looking at the graph, the three algorithms show a clear ranking throughout training. Q-learning pulled ahead earliest - its curve starts climbing noticeably around 50,000 steps and levels off near 0.30 by the halfway point. The key reason is that Q-learning decouples what it *learns* from what it *does*: the update always assumes the agent will act optimally in the future (`max Q(s', a)`), even if the current behavior is still mostly random.

SARSA reached a much more modest ceiling of around 0.11–0.12. It never caught up to Q-learning even after epsilon finished decaying. The gap suggests that the on-policy constraint genuinely hurts here - SARSA's value estimates are pulled down by the exploration noise throughout training, and that distortion partially carries over to the final greedy policy.

REINFORCE barely moved off zero for most of the run, finishing around 0.03–0.04. The algorithm simply doesn't get enough goal-reaching episodes early on to build a useful policy gradient signal. Without seeing the reward, no meaningful update happens.

### Were Q-learning and SARSA different in their behavior?

The graph makes the difference clear: Q-learning climbs steeply in the first 150,000 steps while SARSA's curve is much flatter over the same period, only accelerating once epsilon gets low enough that most actions are greedy anyway.

The reason is how each algorithm treats the next step in its update. Q-learning picks the best action hypothetically - it imagines an agent that always exploits, even during periods of heavy exploration. SARSA picks the action that will actually be taken next, which during early training is often random. So SARSA's Q-values end up reflecting a mix of optimal and exploratory behavior rather than pure optimal behavior. By the end both algorithms use mostly-greedy behavior (ε = 0.05), but the value estimates Q-learning accumulated during training are closer to the true optimal values.

### Was REINFORCE less stable?

REINFORCE's std band across seeds is actually narrow, the five seeds all learned roughly the same slow trajectory. So in terms of cross-seed variance it was the most consistent of the three. The issue is that it consistently learned very little, not that it was unpredictable.

The core problem is timing: REINFORCE only updates after a complete episode ends. On a slippery 8×8 grid with a max episode length of 200 steps, the vast majority of early episodes terminate by falling into a hole with zero reward. The algorithm goes hundreds of thousands of steps receiving almost no training signal. TD methods don't have this problem because they learn from every single step, even when the episode ends badly.

### How did FrozenLake's stochasticity affect learning?

On a non-slippery grid a good policy would succeed close to 100% of the time. Here, the 1/3 intended / 1/3 left / 1/3 right transition model means that even a perfect policy gets unlucky regularly - the agent slides into holes it was actively trying to avoid. This caps V(s₀) at roughly 0.3–0.5 for the optimal policy on this map, which matches what we see in the graph.

Beyond the ceiling effect, the slipperiness also slows down learning. Q-values are averages over many visits to a state-action pair. When the same action from the same state leads to three different next states, those averages take much longer to stabilize. This is particularly painful for REINFORCE: not only is the reward sparse, but the path to the goal is also random, so even successful episodes look very different from each other, making it hard for the policy gradient to identify which actions were actually responsible for the success.
