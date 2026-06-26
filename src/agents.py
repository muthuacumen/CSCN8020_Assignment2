"""Q-Learning agent for the Taxi-v3 environment.

This module contains QLearningAgent — the "brain" of our taxi driver.
The agent learns by building a Q-table: a spreadsheet that scores every
(state, action) pair with an expected future reward.

Think of Q[state, action] like a restaurant rating: the more the agent
visits a (state, action) combo, the more accurate its rating becomes.
After enough visits, the agent knows which action is best in each state.

Mapping to Sutton & Barto (2018) Chapter 6.5 Q-Learning pseudocode
-------------------------------------------------------------------
    Step 1  Initialize Q(s,a) for all s,a       ->  q_table = np.zeros(...)
    Step 3  Choose A via epsilon-greedy from Q   ->  select_action(state)
    Step 5  Q(S,A) <- Q + alpha[R + gamma*maxQ(S') - Q]  ->  update(...)
"""

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent (Sutton & Barto, Section 6.5).

    Learns an action-value function Q(s,a) by applying the TD update rule
    after every step — no need to wait until the end of an episode.
    This is what makes Q-Learning a Temporal Difference (TD) method.

    Parameters
    ----------
    num_states : int
        Total number of states (500 for Taxi-v3).
    num_actions : int
        Total number of actions (6 for Taxi-v3).
    alpha : float
        Learning rate. How much each new experience overwrites old Q values.
        alpha=1.0 -> "believe only the latest experience"; alpha=0 -> "never learn".
    epsilon : float
        Exploration rate. Probability of choosing a random action instead of
        the greedy best. Balances exploration vs exploitation.
    gamma : float
        Discount factor. How much future rewards are worth compared to now.
        gamma=0 -> "only care about immediate reward"; gamma=1 -> patient.
    seed : int
        Random seed for reproducible epsilon-greedy exploration.
    """

    def __init__(self, num_states, num_actions, alpha=0.1, epsilon=0.1,
                 gamma=0.9, seed=42):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # S&B pseudocode Step 1: "Initialize Q(s,a) for all s,a, arbitrarily"
        # Zeros = a neutral start. The agent has no prior opinions about any action.
        self.q_table = np.zeros((num_states, num_actions))

        # Seeded RNG for reproducible random choices during epsilon-greedy
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def select_action(self, state):
        """Epsilon-greedy action selection.

        S&B pseudocode Step 3:
            "Choose A from S using policy derived from Q (e.g., epsilon-greedy)"

        With probability epsilon   -> explore: pick a random action
        With probability 1-epsilon -> exploit: pick argmax_a Q(state, a)

        The np.argmax call over one row of the Q-table is the vectorized
        greedy max — fast and readable. Ties are broken by returning the
        first tied action (lowest action index), a deliberate convention.
        """
        if self.rng.random() < self.epsilon:
            # Explore: try something random — we might discover a better path
            return int(self.rng.integers(self.num_actions))
        # Exploit: use the action we currently think is best
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, terminated):
        """Apply the Q-Learning TD update rule.

        S&B pseudocode Step 5:
            Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]
                                         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                                  TD target

        The quantity  [TD target - Q(S,A)]  is the "TD error" — how surprised
        the agent was by what just happened:
            Positive error -> action was BETTER than expected -> raise Q(S,A)
            Negative error -> action was WORSE than expected  -> lower Q(S,A)

        If the episode just ended (terminated=True), the taxi delivered the
        passenger. There is no future, so gamma * max Q(S') = 0. The target
        is just the immediate reward R.
        """
        if terminated:
            # No future to discount: V(terminal) = 0 by definition
            td_target = reward
        else:
            # Best possible future value from next_state (the "bootstrap" estimate)
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    # ------------------------------------------------------------------
    def greedy_action(self, state):
        """Return the best action with no exploration (epsilon = 0).

        Used during policy EVALUATION — we want to measure how good the
        learned Q-table is without adding noise from random exploration.
        """
        return int(np.argmax(self.q_table[state]))
