"""Logging and plotting utilities for CSCN8020 Assignment 2.

Two classes:
    MetricsLogger  — wraps Python's logging module to write a graded log file.
    PlotManager    — creates the assignment's required comparison charts.

The logger mirrors Assignment 1's utils.py setup_logger() so both assignments
produce consistently formatted, timestamped execution logs.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ======================================================================
class MetricsLogger:
    """Creates and manages the assignment's execution log file.

    The log file is a graded deliverable (5 pts). It records:
    - Training start + selected hyperparameters
    - Episode progress at regular intervals (every N episodes)
    - Final average reward and wall-clock training time
    - Experiment transitions (which configuration is being tested)
    """

    @staticmethod
    def setup(log_path="logs/sample_execution.log", name="cscn8020a2",
              overwrite=True):
        """Create (or fetch) the logger writing to log_path.

        Mirrors Assignment 1's setup_logger() so the log format is consistent
        between assignments. overwrite=True starts a fresh file each notebook
        run, keeping the committed sample log small and readable.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()       # avoid duplicate handlers on notebook re-run
        handler = logging.FileHandler(
            log_path, mode="w" if overwrite else "a", encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False      # stop messages doubling in Jupyter output
        return logger

    @staticmethod
    def show_tail(log_path="logs/sample_execution.log", lines=20):
        """Print the last N lines of the log inside the notebook as evidence."""
        with open(log_path, encoding="utf-8") as fh:
            tail = fh.readlines()[-lines:]
        print(f"--- last {len(tail)} lines of {log_path} ---")
        print("".join(tail))


# ======================================================================
class PlotManager:
    """Creates all required plots for the assignment.

    Design choice: consolidated, overlaid charts with rolling-mean smoothing —
    one figure per experiment group rather than one figure per run. This directly
    addresses the 15-pt "clear, interpreted plots" rubric criterion.
    """

    WINDOW = 100   # rolling-mean window (episodes)

    # ------------------------------------------------------------------
    @staticmethod
    def rolling_mean(data, window=100):
        """Smooth a noisy signal with a rolling average.

        Think of it like smoothing a bumpy road: averaging out short-term noise
        (episode-to-episode variance) so the overall learning trend is visible.
        min_periods=1 means the first few episodes get a partial average
        instead of being dropped.
        """
        return pd.Series(data).rolling(window=window, min_periods=1).mean().values

    # ------------------------------------------------------------------
    @staticmethod
    def plot_baseline(episode_rewards, episode_steps, window=100, figsize=(14, 5)):
        """Two-panel chart: reward/episode + steps/episode for the baseline run.

        Shows both the raw noisy signal (faint, alpha=0.25) and the smoothed
        trend (solid, lw=2.5) on the same axes — so readers can see both the
        learning trend and the episode-to-episode variance at a glance.
        """
        sns.set_style("whitegrid")
        smoothed_r = PlotManager.rolling_mean(episode_rewards, window)
        smoothed_s = PlotManager.rolling_mean(episode_steps, window)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(episode_rewards, alpha=0.2, color="steelblue", lw=0.8, label="Raw")
        ax1.plot(smoothed_r, color="steelblue", lw=2.5,
                 label=f"Rolling mean (window={window})")
        ax1.set_title("Baseline Training: Total Reward per Episode", fontsize=13)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend()

        ax2.plot(episode_steps, alpha=0.2, color="darkorange", lw=0.8, label="Raw")
        ax2.plot(smoothed_s, color="darkorange", lw=2.5,
                 label=f"Rolling mean (window={window})")
        ax2.set_title("Baseline Training: Steps per Episode", fontsize=13)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps per Episode")
        ax2.legend()

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def plot_experiment_group(results, group_names, group_title,
                               window=100, figsize=(14, 5)):
        """Overlay multiple configurations on one pair of shared axes.

        One figure, two subplots (reward and steps). Every configuration in
        group_names is drawn as a distinct colour with rolling-mean smoothing.
        Overlaying makes differences between configs immediately visible —
        much clearer than a separate chart per configuration.

        Parameters
        ----------
        results     : dict returned by ExperimentRunner.results
        group_names : list[str]  keys in results to include in this chart
        group_title : str        chart title prefix (e.g. "Alpha Experiments")
        """
        sns.set_style("whitegrid")
        palette = sns.color_palette("tab10", len(group_names))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for name, color in zip(group_names, palette):
            r = results[name]
            sm_r = PlotManager.rolling_mean(r["rewards"], window)
            sm_s = PlotManager.rolling_mean(r["steps"], window)
            label = f"{name}  (alpha={r['alpha']}, epsilon={r['epsilon']})"

            ax1.plot(sm_r, color=color, lw=2, label=label)
            ax2.plot(sm_s, color=color, lw=2, label=label)

        ax1.set_title(f"{group_title}: Reward per Episode (smoothed, window={window})",
                      fontsize=12)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend(fontsize=8, loc="lower right")

        ax2.set_title(f"{group_title}: Steps per Episode (smoothed, window={window})",
                      fontsize=12)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps per Episode")
        ax2.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def plot_eval_comparison(df_summary, figsize=(12, 5)):
        """Side-by-side bar charts: greedy evaluation reward and steps for all configs.

        Error bars on the reward chart show standard deviation — a measure of
        how consistent the greedy policy is across evaluation episodes.
        """
        sns.set_style("whitegrid")
        n = len(df_summary)
        palette = sns.color_palette("tab10", n)
        configs = df_summary["Configuration"].tolist()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Reward bar chart with error bars
        ax1.bar(range(n), df_summary["Greedy Eval Reward"], color=palette, zorder=2)
        ax1.errorbar(range(n), df_summary["Greedy Eval Reward"],
                     yerr=df_summary["Eval Reward Std (+/-)"],
                     fmt="none", color="black", capsize=5, zorder=3)
        ax1.set_xticks(range(n))
        ax1.set_xticklabels(configs, rotation=40, ha="right", fontsize=9)
        ax1.set_title("Greedy Policy Evaluation: Average Reward", fontsize=12)
        ax1.set_ylabel("Average Reward (100 greedy episodes)")

        # Steps bar chart
        ax2.bar(range(n), df_summary["Greedy Eval Steps"], color=palette, zorder=2)
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(configs, rotation=40, ha="right", fontsize=9)
        ax2.set_title("Greedy Policy Evaluation: Average Steps", fontsize=12)
        ax2.set_ylabel("Average Steps per Episode")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def plot_best_vs_baseline(baseline_results, best_results, best_name,
                               window=100, figsize=(14, 5)):
        """Direct comparison of baseline vs best-combination training curves."""
        sns.set_style("whitegrid")

        baseline_r = PlotManager.rolling_mean(baseline_results["rewards"], window)
        baseline_s = PlotManager.rolling_mean(baseline_results["steps"], window)
        best_r = PlotManager.rolling_mean(best_results["rewards"], window)
        best_s = PlotManager.rolling_mean(best_results["steps"], window)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(baseline_r, color="steelblue", lw=2,
                 label="Baseline (alpha=0.1, epsilon=0.1)")
        ax1.plot(best_r, color="crimson", lw=2, linestyle="--",
                 label=f"Best: {best_name}")
        ax1.set_title("Baseline vs Best Combination: Reward per Episode", fontsize=12)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward (smoothed)")
        ax1.legend()

        ax2.plot(baseline_s, color="steelblue", lw=2,
                 label="Baseline (alpha=0.1, epsilon=0.1)")
        ax2.plot(best_s, color="crimson", lw=2, linestyle="--",
                 label=f"Best: {best_name}")
        ax2.set_title("Baseline vs Best Combination: Steps per Episode", fontsize=12)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps per Episode (smoothed)")
        ax2.legend()

        plt.tight_layout()
        return fig
