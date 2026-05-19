# Programming Assignment 3:
## Comparing Three Algorithms in the FrozenLake 8×8 Environment

In this assignment, you will experimentally compare three reinforcement learning algorithms. The experiments will be conducted in Gymnasium's FrozenLake-v1 environment, sized 8×8, with stochastic dynamics. The goal is to compare the learning speed and quality of the algorithms over time.

| **Algorithm** | **Implementation Type** |
|---|---|
| Q-learning | Tabular |
| SARSA | Tabular |
| REINFORCE | Stochastic tabular policy |

---

## Installing Libraries

Use the following libraries: `gymnasium`, `numpy`, and `matplotlib`.

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Setup

Use the FrozenLake environment sized 8×8 with slipping enabled. Do **not** use the deterministic version of the environment.

```python
import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)
```

The number of states and actions is obtained as follows:

```python
n_states = env.observation_space.n
n_actions = env.action_space.n
```

In FrozenLake 8×8 there are **64 states** and **4 actions**. The initial state is typically state number 0.

---

## Algorithms to Implement

- Tabular Q-learning
- Tabular SARSA
- REINFORCE with a stochastic tabular policy
- **Do not** use neural networks.
- **Do not** use ready-made RL libraries; implement the algorithms yourself.

---

## General Experiment Parameters

Use the following parameters as a starting point for the experiment:

```python
gamma = 0.99
max_steps_per_episode = 200
total_training_steps = 500_000
eval_interval_steps = 10_000
n_eval_episodes = 500
seeds = [0, 1, 2, 3, 4]
```

Stop training for evaluation every 10,000 environment steps, up to a total of 500,000 training steps. Evaluation steps do **not** count as training steps.

---

## Hyperparameters for Q-learning and SARSA

Use the same parameters for both Q-learning and SARSA, unless stated otherwise in the report:

```python
alpha = 0.1
gamma = 0.99

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 300_000
```

It is recommended to use linear epsilon decay based on the number of training steps completed so far:

```python
epsilon = max(
    epsilon_end,
    epsilon_start - step / epsilon_decay_steps * (epsilon_start - epsilon_end)
)
```

---

## REINFORCE Definition

In REINFORCE, use a stochastic tabular policy. The policy parameters will be denoted by `theta`.

> **Important:** `theta` is a table of size (number of states × number of actions). That is, each state–action pair has one parameter.

In FrozenLake 8×8 there are 64 states and 4 actions, therefore:

```python
theta.shape == (64, 4)
```

In general, define:

```python
theta = np.zeros((n_states, n_actions))
```

For FrozenLake 8×8 this is equivalent to:

```python
theta = np.zeros((64, 4))
```

Each value `theta[s, a]` is the weight of action `a` in state `s`. Do not pre-assign fixed values; initialize the table to zero and update it during training. Obtain the policy by applying softmax to the row corresponding to the current state.

```python
def softmax(x):
    x = x - np.max(x)       # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def policy_probs(theta, state):
    return softmax(theta[state])
```

Selecting an action during training:

```python
probs = policy_probs(theta, state)
action = np.random.choice(n_actions, p=probs)
```

Use the following hyperparameters for REINFORCE:

```python
gamma = 0.99
policy_lr = 0.005
use_baseline = False
value_lr = 0.05
theta_init = 0.0
theta_clip = 20.0
```

At each episode, save the trajectory:

```python
states = []
actions = []
rewards = []
```

---

## Performance Metric

Compare the algorithms using a single graph where the x-axis is the total number of training steps completed so far, and the y-axis is the empirical estimate of the initial state value V(s₀).

The value of the initial state must be estimated empirically **only**. Do not rely on the algorithm's internal Q or V values.

At each measurement point, temporarily pause training, freeze the current policy, run it `n_eval_episodes` times from the initial state, and compute the mean return:

```python
estimated_V_s0 = np.mean(eval_returns)
```

---

## Policy Evaluation Procedure

1. Pause training at the designated measurement points.
2. Run the current policy for `n_eval_episodes` episodes.
3. Do **not** update parameters during evaluation.
4. Compute the mean return.
5. Record the pair: (number of training steps so far, empirical value of the initial state).

For Q-learning and SARSA, use a greedy policy during evaluation:

```python
action = np.argmax(Q[state])
```

For REINFORCE, use a greedy policy based on the probabilities obtained from `theta`:

```python
probs = policy_probs(theta, state)
action = np.argmax(probs)
```

Evaluation must always start from the environment's initial state.

---

## Computing the Return During Evaluation

In each evaluation episode, compute the discounted return:

```python
G = 0.0
discount = 1.0

for t in range(max_steps_per_episode):
    # choose action according to the evaluation policy
    next_state, reward, terminated, truncated, info = env.step(action)

    G += discount * reward
    discount *= gamma

    if terminated or truncated:
        break
```

The reported value for the initial state is the average of G across all evaluation episodes.

---

## Required Graph

Produce **a single graph** containing three curves: Q-learning, SARSA, and REINFORCE.

| **Component** | **Definition** |
|---|---|
| x-axis | Number of environment steps |
| y-axis | Estimated value of initial state |
| Curves | Q-learning, SARSA, REINFORCE |
| Measurement points | Every 10,000 training steps |
| Number of evaluation runs | 500 episodes per measurement point |

If running multiple random seeds, display for each algorithm a mean curve and a shaded region showing standard deviation or standard error.

```python
plt.plot(steps, mean_values, label="Q-learning")
plt.fill_between(
    steps,
    mean_values - std_values,
    mean_values + std_values,
    alpha=0.2
)
```

Include a title, legend, and axis labels in the graph.

---

## Counting Training Steps

The x-axis should count environment steps **during training**, not the number of episodes. Each call to `env.step` during training counts as one training step:

```python
next_state, reward, terminated, truncated, info = env.step(action)
```

Evaluation steps do **not** count as training steps.

---

## Episode Termination

An episode ends when:

```python
done = terminated or truncated
```

If the episode has ended, do **not** use a future value in the update.

---

## Reproducibility

Run the experiment over multiple random seeds and report them. For example:

```python
seeds = [0, 1, 2, 3, 4]
```

At the beginning of each run:

```python
obs, info = env.reset(seed=seed)
env.action_space.seed(seed)
np.random.seed(seed)
```

It is recommended to use a separate environment for evaluation, or to ensure that evaluation does not alter the training sequence.

---

## Submission Requirements

1. Complete code for all three algorithms.
2. A single comparison graph with three performance curves.
3. A detailed list of the hyperparameters used.
4. A brief explanation of how V(s₀) was estimated.
5. A short discussion of results: which algorithm learned faster, which achieved better performance, whether Q-learning and SARSA behaved differently, whether REINFORCE was less stable, and how the stochasticity of FrozenLake affected learning.

---

## Important Notes

- Do **not** evaluate performance based on internal Q or V values.
- The required metric is empirical evaluation by running the policy.
- Ensure that all three algorithms are measured at the same time points in terms of number of training steps.
- Use the same FrozenLake environment and the same **seeds** as much as possible across all three algorithms.
- Clearly report any deviation from the recommended parameters.