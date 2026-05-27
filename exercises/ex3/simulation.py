import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from algorithms import q_learning, sarsa, reinforce


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLOT_PATH = os.path.join(HERE, "comparison.png")


def make_env():
    return gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)


def run_experiment(seeds=(0, 1, 2, 3, 4),
                   total_training_steps=500_000,
                   eval_interval_steps=10_000,
                   n_eval_episodes=500,
                   max_steps_per_episode=200,
                   gamma=0.99,
                   plot_path=DEFAULT_PLOT_PATH):

    algos = {
        "Q-learning": q_learning,
        "SARSA": sarsa,
        "REINFORCE": reinforce,
    }

    results = {name: [] for name in algos}
    steps_grid = None

    for name, fn in algos.items():
        print(f"\n=== {name} ===")
        for seed in seeds:
            env = make_env()
            n_states = env.observation_space.n
            n_actions = env.action_space.n

            kwargs = dict(
                gamma=gamma,
                total_training_steps=total_training_steps,
                eval_interval_steps=eval_interval_steps,
                n_eval_episodes=n_eval_episodes,
                max_steps_per_episode=max_steps_per_episode,
                seed=seed,
            )
            steps_log, values_log, _ = fn(env, n_states, n_actions, **kwargs)
            env.close()

            results[name].append(values_log)
            if steps_grid is None:
                steps_grid = steps_log

    print_summary(results, steps_grid)
    plot_results(results, steps_grid, plot_path)
    return results, steps_grid


def print_summary(results, steps_grid):
    """Print mean +/- std of the final V(s0) across seeds, per algorithm."""
    print("\n" + "=" * 60)
    print(f"Final V(s0) summary  (after {steps_grid[-1]:,} training steps)")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 60)
    for name, runs in results.items():
        finals = np.array([r[-1] for r in runs])
        print(f"{name:<12} {finals.mean():>10.4f} {finals.std():>10.4f} "
              f"{finals.min():>10.4f} {finals.max():>10.4f}")
    print("=" * 60)


def plot_results(results, steps_grid, plot_path=DEFAULT_PLOT_PATH):
    plt.figure(figsize=(9, 6))
    for name, runs in results.items():
        min_len = min(len(r) for r in runs)
        arr = np.array([r[:min_len] for r in runs])
        mean_values = arr.mean(axis=0)
        std_values = arr.std(axis=0)
        steps = np.array(steps_grid[:min_len])

        plt.plot(steps, mean_values, label=name)
        plt.fill_between(steps,
                         mean_values - std_values,
                         mean_values + std_values,
                         alpha=0.2)

    plt.xlabel("Number of environment steps")
    plt.ylabel("Estimated value of initial state V(s0)")
    plt.title("FrozenLake 8x8 (slippery): Q-learning vs SARSA vs REINFORCE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved comparison plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment()
