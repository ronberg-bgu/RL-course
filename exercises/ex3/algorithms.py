import numpy as np
import gymnasium as gym


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def policy_probs(theta, state):
    return softmax(theta[state])


def _make_eval_env():
    return gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)


def _evaluate_greedy_Q(Q, eval_env, n_eval_episodes, max_steps, gamma, eval_seed):
    """Greedy evaluation for Q-learning / SARSA. Returns empirical V(s0)."""
    returns = []
    for ep in range(n_eval_episodes):
        state, _ = eval_env.reset(seed=eval_seed + ep)
        G = 0.0
        discount = 1.0
        for _ in range(max_steps):
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            G += discount * reward
            discount *= gamma
            if terminated or truncated:
                break
        returns.append(G)
    return float(np.mean(returns))


def _evaluate_greedy_theta(theta, eval_env, n_eval_episodes, max_steps, gamma, eval_seed):
    """Greedy evaluation for REINFORCE (argmax over softmax probs)."""
    returns = []
    for ep in range(n_eval_episodes):
        state, _ = eval_env.reset(seed=eval_seed + ep)
        G = 0.0
        discount = 1.0
        for _ in range(max_steps):
            probs = policy_probs(theta, state)
            action = int(np.argmax(probs))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            G += discount * reward
            discount *= gamma
            if terminated or truncated:
                break
        returns.append(G)
    return float(np.mean(returns))


def q_learning(env, n_states, n_actions,
               alpha=0.1,
               gamma=0.99,
               epsilon_start=1.0,
               epsilon_end=0.05,
               epsilon_decay_steps=300_000,
               total_training_steps=500_000,
               eval_interval_steps=10_000,
               n_eval_episodes=500,
               max_steps_per_episode=200,
               seed=0):

    np.random.seed(seed)
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)

    eval_env = _make_eval_env()

    Q = np.zeros((n_states, n_actions))
    steps_log = []
    values_log = []
    total_steps = 0
    episode_steps = 0

    while total_steps < total_training_steps:
        epsilon = max(
            epsilon_end,
            epsilon_start - total_steps / epsilon_decay_steps
                          * (epsilon_start - epsilon_end)
        )

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_steps += 1
        done = terminated or truncated or (episode_steps >= max_steps_per_episode)
        total_steps += 1

        best_next = 0.0 if terminated else float(np.max(Q[next_state]))
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state
        if done:
            state, _ = env.reset()
            episode_steps = 0

        if total_steps % eval_interval_steps == 0:
            v_s0 = _evaluate_greedy_Q(Q, eval_env, n_eval_episodes,
                                      max_steps_per_episode, gamma, seed)
            steps_log.append(total_steps)
            values_log.append(v_s0)
            print(f"  [Q-learning seed={seed}] step {total_steps:>7,} | "
                  f"eps={epsilon:.3f} | V(s0)={v_s0:.4f}")

    eval_env.close()
    return steps_log, values_log, Q


def sarsa(env, n_states, n_actions,
          alpha=0.1,
          gamma=0.99,
          epsilon_start=1.0,
          epsilon_end=0.05,
          epsilon_decay_steps=300_000,
          total_training_steps=500_000,
          eval_interval_steps=10_000,
          n_eval_episodes=500,
          max_steps_per_episode=200,
          seed=0):

    np.random.seed(seed)
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)

    eval_env = _make_eval_env()

    Q = np.zeros((n_states, n_actions))
    steps_log = []
    values_log = []
    total_steps = 0
    episode_steps = 0

    def eps_greedy(s, eps):
        if np.random.random() < eps:
            return env.action_space.sample()
        return int(np.argmax(Q[s]))

    epsilon = epsilon_start
    action = eps_greedy(state, epsilon)

    while total_steps < total_training_steps:
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_steps += 1
        done = terminated or truncated or (episode_steps >= max_steps_per_episode)
        total_steps += 1

        epsilon = max(
            epsilon_end,
            epsilon_start - total_steps / epsilon_decay_steps
                          * (epsilon_start - epsilon_end)
        )

        if terminated:
            target = reward
            Q[state, action] += alpha * (target - Q[state, action])
            state, _ = env.reset()
            episode_steps = 0
            action = eps_greedy(state, epsilon)
        else:
            next_action = eps_greedy(next_state, epsilon)
            target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])

            if truncated or episode_steps >= max_steps_per_episode:
                state, _ = env.reset()
                episode_steps = 0
                action = eps_greedy(state, epsilon)
            else:
                state = next_state
                action = next_action

        if total_steps % eval_interval_steps == 0:
            v_s0 = _evaluate_greedy_Q(Q, eval_env, n_eval_episodes,
                                      max_steps_per_episode, gamma, seed)
            steps_log.append(total_steps)
            values_log.append(v_s0)
            print(f"  [SARSA      seed={seed}] step {total_steps:>7,} | "
                  f"eps={epsilon:.3f} | V(s0)={v_s0:.4f}")

    eval_env.close()
    return steps_log, values_log, Q


def reinforce(env, n_states, n_actions,
              gamma=0.99,
              policy_lr=0.005,
              use_baseline=False,
              value_lr=0.05,
              theta_init=0.0,
              theta_clip=20.0,
              total_training_steps=500_000,
              eval_interval_steps=10_000,
              n_eval_episodes=500,
              max_steps_per_episode=200,
              seed=0):

    np.random.seed(seed)
    env.action_space.seed(seed)

    eval_env = _make_eval_env()

    theta = np.full((n_states, n_actions), theta_init, dtype=np.float64)
    V = np.zeros(n_states, dtype=np.float64)  # baseline (only used if use_baseline)

    steps_log = []
    values_log = []
    total_steps = 0
    next_eval = eval_interval_steps

    while total_steps < total_training_steps:
        state, _ = env.reset()
        states, actions, rewards = [], [], []

        for _ in range(max_steps_per_episode):
            probs = policy_probs(theta, state)
            action = int(np.random.choice(n_actions, p=probs))
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_steps += 1

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if terminated or truncated:
                break

            if total_steps >= total_training_steps:
                break

        T = len(rewards)
        G = 0.0
        returns = np.zeros(T, dtype=np.float64)
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        for t in range(T):
            s = states[t]
            a = actions[t]
            Gt = returns[t]

            if use_baseline:
                advantage = Gt - V[s]
                V[s] += value_lr * advantage
            else:
                advantage = Gt

            probs = policy_probs(theta, s)
            grad_log = -probs
            grad_log[a] += 1.0

            theta[s] += policy_lr * advantage * grad_log

        np.clip(theta, -theta_clip, theta_clip, out=theta)

        while total_steps >= next_eval and next_eval <= total_training_steps:
            v_s0 = _evaluate_greedy_theta(theta, eval_env, n_eval_episodes,
                                          max_steps_per_episode, gamma, seed)
            steps_log.append(next_eval)
            values_log.append(v_s0)
            print(f"  [REINFORCE  seed={seed}] step {next_eval:>7,} | "
                  f"V(s0)={v_s0:.4f}")
            next_eval += eval_interval_steps

    eval_env.close()
    return steps_log, values_log, theta
