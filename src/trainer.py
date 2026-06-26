"""Training loop and experiment orchestration for Q-Learning on Taxi-v3.

Two classes:
    QLearningTrainer  — runs one training run (the Week 1 RL loop) and
                        tracks per-episode metrics.
    ExperimentRunner  — orchestrates multiple training runs with different
                        hyperparameter configurations for side-by-side comparison.

The episode loop implemented in QLearningTrainer is the same "reset ->
choose A -> step -> update -> repeat" structure taught in Week 1 with
HelloGymMaze.py, extended with Q-table updates and metric tracking.

Sutton & Barto (2018) Q-Learning pseudocode (Section 6.5) — full mapping:
    Initialize Q(s,a)                       in QLearningAgent.__init__
    Repeat for each episode:
        S <- env.reset()                    state, _ = self.env.reset()
        Repeat for each step of episode:
            A <- epsilon-greedy(Q, S)       agent.select_action(state)
            Take A, observe R, S'           env.step(action)
            Q(S,A) <- Q + alpha[...]        agent.update(...)
            S <- S'
        Until S is terminal or truncated
"""

import time

import gymnasium as gym
import numpy as np
import pandas as pd

from .environments import TaxiEnvironmentManager
from .agents import QLearningAgent


class QLearningTrainer:
    """Runs the Q-Learning episode loop and records per-episode metrics.

    Parameters
    ----------
    env_manager : TaxiEnvironmentManager
        The environment this trainer will interact with.
    agent : QLearningAgent
        The agent whose Q-table will be updated during training.
    logger : logging.Logger or None
        If provided, progress is written to the graded log file.
    """

    def __init__(self, env_manager, agent, logger=None):
        self.env = env_manager
        self.agent = agent
        self.logger = logger

        # Per-episode tracking (empty until training runs)
        self.episode_rewards = []
        self.episode_steps = []
        self.elapsed_seconds = None

    # ------------------------------------------------------------------
    def run(self, num_episodes=5000, log_every=500):
        """Train for num_episodes full episodes, logging every log_every episodes.

        Returns
        -------
        episode_rewards : list[float]
            Total reward collected in each training episode.
        episode_steps : list[int]
            Steps taken in each episode (lower is better once agent learns).
        """
        if self.logger:
            self.logger.info(
                "Training START | alpha=%.4f  epsilon=%.4f  gamma=%.2f  episodes=%d",
                self.agent.alpha, self.agent.epsilon, self.agent.gamma, num_episodes
            )

        start_time = time.perf_counter()

        for episode in range(1, num_episodes + 1):
            # S&B: "S <- initial state of episode"
            state, _ = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                # S&B: "Choose A from S using epsilon-greedy policy derived from Q"
                action = self.agent.select_action(state)

                # S&B: "Take action A, observe R, S'"
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # S&B: "Q(S,A) <- Q(S,A) + alpha[R + gamma*max_a Q(S',a) - Q(S,A)]"
                self.agent.update(state, action, reward, next_state, terminated)

                total_reward += reward
                steps += 1
                state = next_state
                # terminated = passenger delivered successfully
                # truncated  = 200-step cap hit (Taxi-v3 default truncation)
                done = terminated or truncated

            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)

            if self.logger and episode % log_every == 0:
                avg_r = np.mean(self.episode_rewards[-log_every:])
                avg_s = np.mean(self.episode_steps[-log_every:])
                self.logger.info(
                    "Episode %5d/%d | avg_reward(last %d)=%7.2f | avg_steps=%6.1f",
                    episode, num_episodes, log_every, avg_r, avg_s
                )

        self.elapsed_seconds = time.perf_counter() - start_time

        if self.logger:
            self.logger.info(
                "Training DONE | time=%.2fs | final avg_reward(last 500)=%.2f",
                self.elapsed_seconds, np.mean(self.episode_rewards[-500:])
            )

        return self.episode_rewards, self.episode_steps

    # ------------------------------------------------------------------
    def evaluate_greedy(self, num_eval_episodes=100, eval_seed=9999):
        """Evaluate the learned policy with pure greedy action selection (epsilon=0).

        Uses a separate environment instance with a different seed so
        evaluation episodes are distinct from and independent of training.

        Returns
        -------
        avg_reward : float   Mean total reward over num_eval_episodes.
        avg_steps  : float   Mean steps to complete per episode.
        std_reward : float   Standard deviation of episode rewards.
        """
        eval_env = gym.make("Taxi-v3")
        state, _ = eval_env.reset(seed=eval_seed)

        rewards, steps_list = [], []

        for episode in range(num_eval_episodes):
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                # Pure greedy: always pick the action with the highest Q value
                action = self.agent.greedy_action(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
                done = terminated or truncated

            rewards.append(total_reward)
            steps_list.append(steps)

            if episode < num_eval_episodes - 1:
                state, _ = eval_env.reset()

        eval_env.close()
        return float(np.mean(rewards)), float(np.mean(steps_list)), float(np.std(rewards))


# ======================================================================

class ExperimentRunner:
    """Orchestrates multiple Q-Learning runs for hyperparameter comparison.

    Each call to run_single() trains a fresh agent + fresh environment
    so no state leaks between configurations. Results are stored in
    self.results under a descriptive name, then used by PlotManager and
    summary_table() for the Task 3 and Task 4 comparisons.

    Design choice: separate class from QLearningTrainer so each experiment
    is entirely self-contained — testable, readable, and easy to extend.

    Parameters
    ----------
    gamma : float
        Discount factor (fixed across all experiments, as per the assignment).
    num_episodes : int
        Training episodes per run.
    seed : int
        Random seed (same seed for every run ensures fair comparison).
    logger : logging.Logger or None
        Shared logger that records all experiment transitions.
    """

    def __init__(self, gamma=0.9, num_episodes=5000, seed=42, logger=None):
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.seed = seed
        self.logger = logger
        self.results = {}

    # ------------------------------------------------------------------
    def run_single(self, alpha, epsilon, name):
        """Train one configuration and store results under name.

        Parameters
        ----------
        alpha   : float   Learning rate for this run.
        epsilon : float   Exploration rate for this run.
        name    : str     Descriptive key for this configuration.
        """
        if self.logger:
            self.logger.info(
                "=== Experiment: '%s' | alpha=%.4f  epsilon=%.4f  gamma=%.2f ===",
                name, alpha, epsilon, self.gamma
            )
        print(f"  Running '{name}'  (alpha={alpha}, epsilon={epsilon}) ...",
              end=" ", flush=True)

        env_manager = TaxiEnvironmentManager(seed=self.seed)
        agent = QLearningAgent(
            num_states=TaxiEnvironmentManager.NUM_STATES,
            num_actions=TaxiEnvironmentManager.NUM_ACTIONS,
            alpha=alpha, epsilon=epsilon, gamma=self.gamma,
            seed=self.seed
        )
        trainer = QLearningTrainer(env_manager, agent, logger=self.logger)
        rewards, steps = trainer.run(self.num_episodes, log_every=500)

        avg_eval_reward, avg_eval_steps, std_eval = trainer.evaluate_greedy()

        self.results[name] = {
            "alpha": alpha,
            "epsilon": epsilon,
            "rewards": rewards,
            "steps": steps,
            "eval_reward": avg_eval_reward,
            "eval_steps": avg_eval_steps,
            "eval_std": std_eval,
        }

        env_manager.close()
        print(f"done.  Greedy eval: reward={avg_eval_reward:.1f} (+-{std_eval:.1f})  "
              f"steps={avg_eval_steps:.1f}")
        return rewards, steps

    # ------------------------------------------------------------------
    def summary_table(self):
        """Return (DataFrame, styled DataFrame) comparing all configurations.

        The best evaluation reward is highlighted green; the fewest steps blue.
        """
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Configuration": name,
                "alpha": r["alpha"],
                "epsilon": r["epsilon"],
                "gamma": self.gamma,
                "Greedy Eval Reward": round(r["eval_reward"], 2),
                "Eval Reward Std (+/-)": round(r["eval_std"], 2),
                "Greedy Eval Steps": round(r["eval_steps"], 1),
            })
        df = pd.DataFrame(rows)

        styled = (
            df.style
            .highlight_max(subset=["Greedy Eval Reward"], color="lightgreen")
            .highlight_min(subset=["Greedy Eval Steps"], color="lightblue")
            .format({
                "alpha": "{:.3f}", "epsilon": "{:.3f}", "gamma": "{:.2f}",
                "Greedy Eval Reward": "{:.2f}",
                "Eval Reward Std (+/-)": "{:.2f}",
                "Greedy Eval Steps": "{:.1f}",
            })
        )
        return df, styled
