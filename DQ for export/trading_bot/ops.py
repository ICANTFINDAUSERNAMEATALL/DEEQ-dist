import math

import tensorflow as tf
import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)

def get_state(data, t, n_days, inventory, esmad, dd_trades):
    """Returns an n-day state representation ending at time t
    THIS IS ALSO THE DATA THAT GETS PUSHED INTO THE MEMORY OF THE THING. 
    I WILL NEED TO MODIFY THIS TO PROCESS DIFFERENT DATA, NOT JUST CHANGE IN PRICING.
    inventory should be a 1 for long, 2 for short, 0 for none
    """


    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []

    for i in range(n_days - 1):
        ll = [sigmoid((i-j)) for i, j in zip(block[i + 1], block[i])][:-1] + [sigmoid(((block[i+1][-1] - block[i][-1]) / 75))]
        ll2 = [sigmoid(esmad[0][i]), sigmoid(esmad[1][i])]
        res.append(ll + ll2)
    
    # mas = [[sigmoid(i)] for i in esmad]
    # invs = [inventory] # REMOVING THIS FOR NOW

    return np.array([res]), np.array([inventory, dd_trades])

def get_invs_state(inventory, dd_trades):
    return np.array([inventory, dd_trades])

# test commit


def create_rl_dataset_pipeline(
    states0,  # NumPy array of current states (shape: [memory_size, state_dim1, state_dim2]) - adjust dimensions
    states1,  # NumPy array of states1 features (if applicable, shape: [memory_size, state_dim1, state_dim2]) - adjust dimensions
    actions,  # NumPy array of actions (shape: [memory_size,])
    rewards,  # NumPy array of rewards (shape: [memory_size,])
    next_states0, # NumPy array of next states0 features (shape: [memory_size, state_dim1, state_dim2]) - adjust dimensions
    next_states1, # NumPy array of next states1 features (shape: [memory_size, state_dim1, state_dim2]) - adjust dimensions
    dones,    # NumPy array of done flags (shape: [memory_size,])
    batch_size,
    buffer_size=None, # Buffer size for shuffling (if shuffling is needed)
    normalize_states=False, # Flag to enable/disable state normalization
    state0_mean=None,     # Pre-calculated mean for state0 normalization (if normalize_states=True)
    state0_std=None,      # Pre-calculated std for state0 normalization (if normalize_states=True)
    state1_mean=None,     # Pre-calculated mean for state1 normalization (if normalize_states=True)
    state1_std=None       # Pre-calculated std for state1 normalization (if normalize_states=True)
):
    """
    Creates a tf.data.Dataset pipeline for RL experience tuples.

    Args:
        states0: NumPy array of current states (first input to model).
        states1: NumPy array of states1 features (second input to model).
        actions: NumPy array of actions.
        rewards: NumPy array of rewards.
        next_states0: NumPy array of next states0 features.
        next_states1: NumPy array of next states1 features.
        dones: NumPy array of done flags.
        batch_size: The batch size for training.
        buffer_size: Buffer size for shuffling (if shuffling, e.g., replay buffer size).
        normalize_states: Boolean flag to enable/disable state normalization.
        state0_mean: Mean for state0 normalization (pre-calculated).
        state0_std: Std dev for state0 normalization (pre-calculated).
        state1_mean: Mean for state1 normalization (pre-calculated).
        state1_std: Std dev for state1 normalization (pre-calculated).

    Returns:
        A tf.data.Dataset object.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (states0, states1, actions, rewards, next_states0, next_states1, dones)
    )

    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    def preprocess_experience(state0, state1, action, reward, next_state0, next_state1, done):
        """Preprocessing function to be applied to each experience tuple."""

        processed_state0 = tf.cast(state0, dtype=tf.float32)
        processed_state1 = tf.cast(state1, dtype=tf.float32)
        processed_action = tf.cast(action, dtype=tf.int32)
        processed_reward = tf.cast(reward, dtype=tf.float32)
        processed_next_state0 = tf.cast(next_state0, dtype=tf.float32)
        processed_next_state1 = tf.cast(next_state1, dtype=tf.float32)
        processed_done = tf.cast(done, dtype=tf.float32) # Use float32 for done flag if needed in loss

        if normalize_states:
            if state0_mean is not None and state0_std is not None:
                processed_state0 = (processed_state0 - state0_mean) / (state0_std + 1e-7) # Add small epsilon to avoid division by zero
            if state1_mean is not None and state1_std is not None:
                processed_state1 = (processed_state1 - state1_mean) / (state1_std + 1e-7)
            if state0_mean is not None and state0_std is not None:
                processed_next_state0 = (processed_next_state0 - state0_mean) / (state0_std + 1e-7) # Use state0 stats for next_state0 as well? Adjust if needed
            if state1_mean is not None and state1_std is not None:
                processed_next_state1 = (processed_next_state1 - state1_mean) / (state1_std + 1e-7)


        return (processed_state0, processed_state1, processed_action, processed_reward,
                processed_next_state0, processed_next_state1, processed_done)

    dataset = dataset.map(preprocess_experience, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Batch and Prefetch

    return dataset
