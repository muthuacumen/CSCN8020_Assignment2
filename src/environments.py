"""Taxi-v3 environment wrapper for CSCN8020 Assignment 2.

The Taxi-v3 environment ships with Gymnasium. This module wraps it inside
TaxiEnvironmentManager, which documents the state encoding, action meanings,
and reward structure in one place — so notebook cells that USE the environment
stay clean and readable.

Think of TaxiEnvironmentManager as a "tour guide" for Taxi-v3: it knows all
the rules, explains them clearly, and hands control over to the agent.

Course-code lineage
-------------------
Follows the gymnasium.Env reset()/step() pattern from
HelloGymMaze/helloGymMaze.py (Week 1 RL loop) and the Taxi example
introduced in CSCN8020 lecture materials.
"""

import gymnasium as gym
import numpy as np


class TaxiEnvironmentManager:
    """Manages the Taxi-v3 Gymnasium environment.

    Taxi-v3 at a glance
    -------------------
    A 5×5 city grid. The taxi starts at a random position. A passenger waits
    at one of four pickup spots (R, G, Y, B). Goal: drive to the passenger,
    pick them up, drive to the destination, and drop them off.

    State encoding  (500 states total)
    ------------------------------------
    Each state is a single integer derived by:
        state = ((row * 5 + col) * 5 + passenger_location) * 4 + destination
    where:
        row, col            in {0,1,2,3,4}  — taxi position   (25 combos)
        passenger_location  in {0=R, 1=G, 2=Y, 3=B, 4=in_taxi}
        destination         in {0=R, 1=G, 2=Y, 3=B}
    Total: 25 x 5 x 4 = 500 states

    Action space  (6 discrete actions)
    ------------------------------------
        0 = South   1 = North   2 = East   3 = West
        4 = Pickup  5 = Dropoff

    Reward structure
    ----------------
        +20  successful passenger delivery
         -1  every step taken (time penalty — keeps the agent efficient)
        -10  illegal Pickup or Dropoff attempt (wrong location)
    """

    NUM_STATES = 500
    NUM_ACTIONS = 6
    ACTION_NAMES = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    LOCATION_NAMES = ["Red (R)", "Green (G)", "Yellow (Y)", "Blue (B)", "In-Taxi"]

    def __init__(self, seed=42):
        """Create the Taxi-v3 environment and remember the reproducibility seed."""
        self._seed = seed
        self._episode_count = 0
        self.env = gym.make("Taxi-v3")

    # ------------------------------------------------------------------
    # Gymnasium API pass-through
    # ------------------------------------------------------------------
    def reset(self):
        """Start a new episode. Seeds the RNG only on the very first call.

        Seeding once at the start means the entire training run is reproducible
        from seed=42 without repeating the same episode every time.
        """
        if self._episode_count == 0:
            state, info = self.env.reset(seed=self._seed)
        else:
            state, info = self.env.reset()
        self._episode_count += 1
        return int(state), info

    def step(self, action):
        """Take one step. Returns (next_state, reward, terminated, truncated, info)."""
        next_state, reward, terminated, truncated, info = self.env.step(int(action))
        return int(next_state), float(reward), terminated, truncated, info

    def close(self):
        """Release the environment's resources when we are done."""
        self.env.close()

    # ------------------------------------------------------------------
    # State decoding helpers
    # ------------------------------------------------------------------
    def decode_state(self, state):
        """Decode an integer state into (row, col, passenger_location, destination).

        Reverses the encoding formula:
            state = ((row*5 + col)*5 + pass_loc)*4 + dest
        by peeling off the last factor first (modulo then integer divide).
        """
        dest = state % 4
        state //= 4
        pass_loc = state % 5
        state //= 5
        col = state % 5
        row = state // 5
        return row, col, pass_loc, dest

    def describe_state(self, state):
        """Return a human-readable description of a flat state integer."""
        row, col, pass_loc, dest = self.decode_state(state)
        return (
            f"Taxi at (row={row}, col={col})  |  "
            f"Passenger: {self.LOCATION_NAMES[pass_loc]}  |  "
            f"Destination: {self.LOCATION_NAMES[dest]}"
        )
