"""Exercise 3 scaffolding for FrozenLake 8x8 algorithm comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExperimentConfig:
    """Holds hyperparameters and runtime settings for the experiments."""

    gamma: float = 0.99
    max_steps_per_episode: int = 200
    total_training_steps: int = 500_000
    eval_interval_steps: int = 10_000
    n_eval_episodes: int = 500
    seeds: List[int] = None

    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000

    policy_lr: float = 0.005
    use_baseline: bool = False
    value_lr: float = 0.05
    theta_init: float = 0.0
    theta_clip: float = 20.0


@dataclass
class EvalSeries:
    """Stores evaluation results over training steps for one run.

    Attributes:
        steps: Training step counts where evaluation occurred.
        values: Empirical estimates of V(s0) at each step.
    """

    steps: List[int]
    values: List[float]


def make_env(seed: int | None = None) -> gym.Env:
    """Create the FrozenLake 8x8 environment with stochastic dynamics.

    Args:
        seed: Optional seed for environment reset.

    Returns:
        A Gymnasium FrozenLake-v1 environment.
    """
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    if seed is not None:
        env.reset(seed=seed)
    return env


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: The seed to use for numpy RNG.
    """
    np.random.seed(seed)


def epsilon_by_step(step: int, config: ExperimentConfig) -> float:
    """Compute epsilon for epsilon-greedy policies using linear decay.

    Args:
        step: Current training step.
        config: ExperimentConfig with epsilon parameters.

    Returns:
        Epsilon value for the given training step.
    """
    return max(
        config.epsilon_end,
        config.epsilon_start - (step / config.epsilon_decay_steps) * (config.epsilon_start - config.epsilon_end),
    )


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax.

    Args:
        x: 1D array of logits.

    Returns:
        1D array of probabilities.
    """
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def policy_probs(theta: np.ndarray, state: int) -> np.ndarray:
    """Return action probabilities for a state under a tabular softmax policy.

    Args:
        theta: Parameter table of shape (n_states, n_actions).
        state: Current state index.

    Returns:
        Action probabilities for the given state.
    """
    return softmax(theta[state])


def evaluate_greedy_q(
    env: gym.Env,
    q_table: np.ndarray,
    config: ExperimentConfig,
) -> float:
    """Evaluate a greedy policy derived from a Q-table.

    Args:
        env: Evaluation environment (no learning).
        q_table: Q-values table of shape (n_states, n_actions).
        config: ExperimentConfig with gamma and max_steps_per_episode.

    Returns:
        Empirical estimate of V(s0) averaged over evaluation episodes.
    """
    returns = []
    for _ in range(config.n_eval_episodes):
        state, _ = env.reset()
        g = 0.0
        discount = 1.0
        for _ in range(config.max_steps_per_episode):
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            g += discount * reward
            discount *= config.gamma
            if terminated or truncated:
                break
            state = next_state
        returns.append(g)
    return float(np.mean(returns))


def evaluate_greedy_policy(
    env: gym.Env,
    policy_fn: Callable[[int], int],
    config: ExperimentConfig,
) -> float:
    """Evaluate a greedy policy provided as a callable.

    Args:
        env: Evaluation environment (no learning).
        policy_fn: Function mapping state -> action.
        config: ExperimentConfig with gamma and max_steps_per_episode.

    Returns:
        Empirical estimate of V(s0) averaged over evaluation episodes.
    """
    returns = []
    for _ in range(config.n_eval_episodes):
        state, _ = env.reset()
        g = 0.0
        discount = 1.0
        for _ in range(config.max_steps_per_episode):
            action = int(policy_fn(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            g += discount * reward
            discount *= config.gamma
            if terminated or truncated:
                break
            state = next_state
        returns.append(g)
    return float(np.mean(returns))


def q_learning(
    env: gym.Env,
    eval_env: gym.Env,
    config: ExperimentConfig,
) -> EvalSeries:
    """Run tabular Q-learning with periodic evaluation.

    Args:
        env: Training environment.
        eval_env: Separate environment for evaluation.
        config: ExperimentConfig with learning and evaluation settings.

    Returns:
        EvalSeries containing evaluation steps and values.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=float)

    eval_steps: List[int] = []
    eval_values: List[float] = []

    training_steps = 0
    state, _ = env.reset()

    while training_steps < config.total_training_steps:
        epsilon = epsilon_by_step(training_steps, config)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        best_next = 0.0 if done else np.max(q_table[next_state])
        td_target = reward + config.gamma * best_next
        q_table[state, action] += config.alpha * (td_target - q_table[state, action])

        training_steps += 1

        if training_steps % config.eval_interval_steps == 0:
            value = evaluate_greedy_q(eval_env, q_table, config)
            eval_steps.append(training_steps)
            eval_values.append(value)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    return EvalSeries(steps=eval_steps, values=eval_values)


def sarsa(
    env: gym.Env,
    eval_env: gym.Env,
    config: ExperimentConfig,
) -> EvalSeries:
    """Run tabular SARSA with periodic evaluation.

    Args:
        env: Training environment.
        eval_env: Separate environment for evaluation.
        config: ExperimentConfig with learning and evaluation settings.

    Returns:
        EvalSeries containing evaluation steps and values.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=float)

    eval_steps: List[int] = []
    eval_values: List[float] = []

    training_steps = 0
    state, _ = env.reset()

    epsilon = epsilon_by_step(training_steps, config)
    action = (
        env.action_space.sample()
        if np.random.rand() < epsilon
        else int(np.argmax(q_table[state]))
    )

    while training_steps < config.total_training_steps:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            td_target = reward
            next_action = None
        else:
            epsilon = epsilon_by_step(training_steps, config)
            next_action = (
                env.action_space.sample()
                if np.random.rand() < epsilon
                else int(np.argmax(q_table[next_state]))
            )
            td_target = reward + config.gamma * q_table[next_state, next_action]

        q_table[state, action] += config.alpha * (td_target - q_table[state, action])

        training_steps += 1

        if training_steps % config.eval_interval_steps == 0:
            value = evaluate_greedy_q(eval_env, q_table, config)
            eval_steps.append(training_steps)
            eval_values.append(value)

        if done:
            state, _ = env.reset()
            epsilon = epsilon_by_step(training_steps, config)
            action = (
                env.action_space.sample()
                if np.random.rand() < epsilon
                else int(np.argmax(q_table[state]))
            )
        else:
            state = next_state
            action = int(next_action)

    return EvalSeries(steps=eval_steps, values=eval_values)


def reinforce(
    env: gym.Env,
    eval_env: gym.Env,
    config: ExperimentConfig,
) -> EvalSeries:
    """Run REINFORCE with a tabular softmax policy and periodic evaluation.

    Args:
        env: Training environment.
        eval_env: Separate environment for evaluation.
        config: ExperimentConfig with learning and evaluation settings.

    Returns:
        EvalSeries containing evaluation steps and values.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    theta = np.full((n_states, n_actions), config.theta_init, dtype=float)

    eval_steps: List[int] = []
    eval_values: List[float] = []

    training_steps = 0

    while training_steps < config.total_training_steps:
        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []

        state, _ = env.reset()

        for _ in range(config.max_steps_per_episode):
            probs = policy_probs(theta, state)
            action = int(np.random.choice(n_actions, p=probs))

            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            training_steps += 1

            if training_steps % config.eval_interval_steps == 0:
                policy_fn = lambda s: int(np.argmax(policy_probs(theta, s)))
                value = evaluate_greedy_policy(eval_env, policy_fn, config)
                eval_steps.append(training_steps)
                eval_values.append(value)

            if training_steps >= config.total_training_steps:
                break

            if terminated or truncated:
                break

            state = next_state

        if len(rewards) == 0:
            continue

        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + config.gamma * g
            returns.append(g)
        returns.reverse()

        for t, (state, action, g_t) in enumerate(zip(states, actions, returns)):
            probs = policy_probs(theta, state)
            grad_log = -probs
            grad_log[action] += 1.0

            baseline = 0.0
            if config.use_baseline:
                baseline = 0.0

            advantage = g_t - baseline
            theta[state] += config.policy_lr * advantage * grad_log
            theta[state] = np.clip(theta[state], -config.theta_clip, config.theta_clip)

    return EvalSeries(steps=eval_steps, values=eval_values)


def aggregate_runs(series_list: List[EvalSeries]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Aggregate evaluation series over multiple seeds.

    Args:
        series_list: List of EvalSeries from different seeds.

    Returns:
        Tuple of (steps, mean_values, std_values).
    """
    steps = series_list[0].steps
    values = np.array([s.values for s in series_list], dtype=float)
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)
    return steps, mean_values, std_values


def plot_results(results: Dict[str, Tuple[List[int], np.ndarray, np.ndarray]]) -> None:
    """Plot mean and standard deviation curves for each algorithm.

    Args:
        results: Mapping from algorithm name to (steps, mean, std).
    """
    plt.figure(figsize=(10, 6))
    for name, (steps, mean_values, std_values) in results.items():
        plt.plot(steps, mean_values, label=name)
        plt.fill_between(
            steps,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
        )

    plt.title("FrozenLake 8x8: Q-learning vs SARSA vs REINFORCE")
    plt.xlabel("Training steps")
    plt.ylabel("Estimated V(s0)")
    plt.legend()
    plt.tight_layout()


def run_experiment(config: ExperimentConfig) -> None:
    """Run all algorithms across seeds and plot a comparison graph.

    Args:
        config: ExperimentConfig with hyperparameters and settings.
    """
    if config.seeds is None:
        config.seeds = [0, 1, 2, 3, 4]

    q_runs: List[EvalSeries] = []
    sarsa_runs: List[EvalSeries] = []
    reinforce_runs: List[EvalSeries] = []

    for seed in config.seeds:
        set_global_seed(seed)

        env = make_env(seed)
        eval_env = make_env(seed + 10_000)

        env.action_space.seed(seed)
        eval_env.action_space.seed(seed + 10_000)

        q_runs.append(q_learning(env, eval_env, config))

        env = make_env(seed)
        eval_env = make_env(seed + 10_000)
        env.action_space.seed(seed)
        eval_env.action_space.seed(seed + 10_000)

        sarsa_runs.append(sarsa(env, eval_env, config))

        env = make_env(seed)
        eval_env = make_env(seed + 10_000)
        env.action_space.seed(seed)
        eval_env.action_space.seed(seed + 10_000)

        reinforce_runs.append(reinforce(env, eval_env, config))

    results = {
        "Q-learning": aggregate_runs(q_runs),
        "SARSA": aggregate_runs(sarsa_runs),
        "REINFORCE": aggregate_runs(reinforce_runs),
    }

    plot_results(results)


def main() -> None:
    """Entry point for running the experiment from the command line."""
    config = ExperimentConfig()
    run_experiment(config)
    plt.show()


if __name__ == "__main__":
    main()
