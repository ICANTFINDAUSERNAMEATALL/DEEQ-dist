import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K
import sys
import os
from math import pow

from keras.models import Sequential, Model
from keras.models import load_model, clone_model
from keras.layers import (
    Dense, 
    Flatten, 
    Reshape, 
    Dropout, 
    Conv1D, 
    BatchNormalization, 
    MaxPooling1D, 
    concatenate, 
    LSTM,
    SeparableConv1D,
    Attention,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Bidirectional,
    GRU,
    AveragePooling1D,
    TimeDistributed,
    Lambda,
    GaussianNoise
)

import keras as kkeras
from trading_bot.ops import sigmoid, create_rl_dataset_pipeline
# from keras.metrics import Accuracy
from keras.activations import selu
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras.callbacks import LearningRateScheduler
from trading_bot.methods import evaluate_model
import device_vars as dv

def huber_loss(y_true, y_pred, clip_delta=25.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss)) # * 100


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=50, pretrained=False, model_name=None, ptm=None, mem_len=100, ma_state_size = 15):
        self.episode_number = 1
        self.strategy = strategy
        self.iternum = 0
        self.return_zero = False
        self.inventoryNorm = []
        self.i2Norm = []
        self.inventoryInv = []
        self.i2Inv = []
        self.i2 = []
        self.inventory = []

        self.memory_length = mem_len
        self.memory_states0 = np.empty((mem_len, state_size, 7), dtype=np.float16) # Pre-allocate NumPy arrays
        self.memory_states1 = np.empty((mem_len, 2), dtype=np.float16) # Adjust state_size appropriately
        self.memory_actions = np.empty(mem_len, dtype=np.int32)
        self.memory_rewards = np.empty(mem_len, dtype=np.float16)
        self.memory_next_states0 = np.empty((mem_len, state_size, 7), dtype=np.float16)
        self.memory_next_states1_noact = np.empty((mem_len, 2), dtype=np.float16)
        self.memory_next_states1_act_win  = np.empty((mem_len,2), dtype=np.float16)
        self.memory_next_states1_act_loss = np.empty((mem_len,2), dtype=np.float16)
        self.memory_dones = np.empty(mem_len, dtype=np.float16) # Or dtype=np.bool_
        self.memory_index = 0 # Keep track of current index in memory
        self.memory_count = 0 # Keep track of how many experiences are stored

        self.set_a_indicies = []
        self.set_b_indicies = []

        self.X_train_cd  = deque(maxlen=mem_len)
        self.X_train_mav = deque(maxlen=mem_len)
        self.y_train     = deque(maxlen=mem_len)
        
        # agent config
        self.state_size = state_size    	     # normalized previous days
        self.ma_state_size = ma_state_size * 2 + 1   # MA state size, also includes inventory
        self.action_size = 3           		     # [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory_length = mem_len
        self.memory = deque(maxlen=self.memory_length)  # Changing to 25 million from 50k due to amount of data
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.9 # affinity for long term reward
        self.epsilon = 0.99
        self.epsilon_min = 0.025
        self.epsilon_decay = 0.9998 # Could modify this decay amount, since the data may be longer. It reaches the min after less than 1000 trials
        self.learning_rate_initial = 0.00025 # if dv.train_1_day == False else 0.001
        self.learning_rate_decay = 0.75
        # self.learning_rate_min = 0.00001
        self.learning_rate = self.learning_rate_initial
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}
        self.optimizer = Adam(learning_rate=self.learning_rate_sched, clipnorm=1)

        if ptm:
            self.model = self.load(ptm)
        elif pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"] and not pretrained:
            self.n_iter = 1
            self.reset_every = reset_every
            self.tau_value = 0.0015

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def reset_mem(self):
        mem_len = self.memory_length
        state_size = self.state_size
        ma_state_size = self.ma_state_size

        self.memory_states0 = np.empty((mem_len, state_size, 5), dtype=np.float32) # Pre-allocate NumPy arrays
        self.memory_states1 = np.empty((mem_len, ma_state_size * 2 + 1), dtype=np.float32) # Adjust state_size appropriately
        self.memory_actions = np.empty(mem_len, dtype=np.int32)
        self.memory_rewards = np.empty(mem_len, dtype=np.float32)
        self.memory_next_states0 = np.empty((mem_len, state_size, 5), dtype=np.float32)
        self.memory_next_states1_noact = np.empty((mem_len, ma_state_size * 2 + 1), dtype=np.float32)
        self.memory_next_states1_act = np.empty((mem_len, ma_state_size * 2 + 1), dtype=np.float32)
        self.memory_dones = np.empty(mem_len, dtype=np.float32) # Or dtype=np.bool_
        self.memory_index = 0 # Keep track of current index in memory
        self.memory_count = 0 # Keep track of how many experiences are stored

        self.set_a_indicies = []
        self.set_b_indicies = []

        self.X_train_cd  = deque(maxlen=mem_len)
        self.X_train_mav = deque(maxlen=mem_len)
        self.y_train     = deque(maxlen=mem_len)

        self.memory = deque(maxlen=self.memory_length)

        self.epsilon = 0.99
        
        return True

    def learning_rate_sched(self):
        return self.learning_rate_initial * pow(self.learning_rate_decay, self.episode_number)

    # NEW MODEL, TESTING
    def _model(self):
        """Creates the model
        """
        self.num_heads_mha = 4 # 8
        self.key_dim_mha = 16  # 32

        self.light_dropout_rate   = 0.1
        self.heavy_dropout_rate   = 0.15

        self.gauss_rate_inp  = 0.2
        self.gauss_rate_norm = 0.1

        self.l1l2 = L1L2(l1 = 0.0025, l2 = 0.0075)

        candle_inputs = kkeras.Input(shape=(self.state_size, 7), name="candle_input") # shape=(30, 5)
        inv_state = kkeras.Input(shape=(2), name="inv_state") # shape=(61, )

        ### MA Input Branch (Simplified)
        position_size_processed = Dense(8, activation='leaky_relu', kernel_regularizer=self.l1l2)(inv_state) # Simpler Dense layer
        
        ### CNN Branch (CNN) 
        ### BLOCK 1
        cnn_branch = GaussianNoise(self.gauss_rate_inp)(candle_inputs)
        cnn_branch = Conv1D(filters=128, kernel_size=3, activation='leaky_relu', kernel_regularizer=self.l1l2)(cnn_branch) # Fewer filters
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = AveragePooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Dropout(self.light_dropout_rate)(cnn_branch)
        cnn_branch = GaussianNoise(self.gauss_rate_norm)(cnn_branch)
        cnn_branch = MultiHeadAttention(num_heads=self.num_heads_mha, key_dim=self.key_dim_mha)(cnn_branch, cnn_branch, cnn_branch)

        ### BLOCK 2
        cnn_branch = Conv1D(filters=192, kernel_size=3, activation='leaky_relu', kernel_regularizer=self.l1l2)(cnn_branch) # Still some CNN capacity
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = AveragePooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Dropout(self.light_dropout_rate)(cnn_branch)
        cnn_branch = GaussianNoise(self.gauss_rate_norm)(cnn_branch)
        cnn_branch = MultiHeadAttention(num_heads=self.num_heads_mha, key_dim=self.key_dim_mha)(cnn_branch, cnn_branch, cnn_branch)

        ### BLOCK 3
        cnn_branch = Conv1D(filters=256, kernel_size=3, activation='leaky_relu', kernel_regularizer=self.l1l2)(cnn_branch) # Still some CNN capacity
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = AveragePooling1D(pool_size=3)(cnn_branch)
        cnn_branch = Dropout(self.heavy_dropout_rate)(cnn_branch)
        cnn_branch = GaussianNoise(self.gauss_rate_norm)(cnn_branch)
        cnn_branch = MultiHeadAttention(num_heads=self.num_heads_mha, key_dim=self.key_dim_mha)(cnn_branch, cnn_branch, cnn_branch)

        cnn_branch = Flatten()(cnn_branch)

        # ### RNN Branch (LSTM) or (GRU)
        per_candle_branch = Lambda(lambda x: x[:, -5:, :])(candle_inputs)
        per_candle_branch = GaussianNoise(self.gauss_rate_inp)(per_candle_branch)
        per_candle_branch = TimeDistributed(Dense(32, activation='leaky_relu', kernel_regularizer=self.l1l2))(per_candle_branch)
        per_candle_branch = TimeDistributed(Dense(32, activation='leaky_relu', kernel_regularizer=self.l1l2))(per_candle_branch)
        per_candle_branch = TimeDistributed(Dense(16, activation='leaky_relu', kernel_regularizer=self.l1l2))(per_candle_branch) # Process each candle's 7 features with a Dense layer

        per_candle_branch = Conv1D(filters=32, kernel_size=3, activation='leaky_relu', kernel_regularizer=self.l1l2)(per_candle_branch) # Still some CNN capacity
        per_candle_branch = BatchNormalization()(per_candle_branch)
        per_candle_branch = AveragePooling1D(pool_size = 2)(per_candle_branch)
        per_candle_branch = Dropout(self.light_dropout_rate)(per_candle_branch)
        per_candle_branch = GaussianNoise(self.gauss_rate_norm)(per_candle_branch)
        per_candle_branch = MultiHeadAttention(num_heads=self.num_heads_mha, key_dim=self.key_dim_mha)(per_candle_branch, per_candle_branch, per_candle_branch)

        per_candle_branch = Flatten()(per_candle_branch) # Flatten the output to combine processed candles into a vector

        ### Combine Branches
        combined_features = concatenate([cnn_branch, per_candle_branch, position_size_processed])

        ### Dense Layers (Simplified) 384
        dense_branch = Dropout(self.heavy_dropout_rate)(combined_features)
        dense_branch = Dense(512, activation='leaky_relu', kernel_regularizer=self.l1l2)(dense_branch) # Fewer Dense layers, moderate size
        dense_branch = BatchNormalization()(dense_branch)
        dense_branch = Attention()([dense_branch, dense_branch])

        dense_branch = concatenate([dense_branch, position_size_processed])

        dense_branch = Dropout(self.heavy_dropout_rate)(combined_features)
        dense_branch = Dense(768, activation='leaky_relu', kernel_regularizer=self.l1l2)(dense_branch) # Fewer Dense layers, moderate size
        dense_branch = BatchNormalization()(dense_branch)
        dense_branch = Attention()([dense_branch, dense_branch])

        dense_branch = concatenate([dense_branch, position_size_processed])

        dense_branch = Dropout(self.heavy_dropout_rate)(dense_branch)
        dense_branch = Dense(768, activation='leaky_relu', kernel_regularizer=self.l1l2)(dense_branch) # Fewer Dense layers, moderate size
        dense_branch = BatchNormalization()(dense_branch)
        dense_branch = Attention()([dense_branch, dense_branch])

        dense_branch = concatenate([dense_branch, position_size_processed])

        dense_branch = Dropout(self.heavy_dropout_rate)(dense_branch)
        dense_branch = Dense(512, activation='leaky_relu', kernel_regularizer=self.l1l2)(dense_branch) # Fewer Dense layers, moderate size
        dense_branch = BatchNormalization()(dense_branch)
        dense_branch = Attention()([dense_branch, dense_branch])

        dense_branch = concatenate([dense_branch, position_size_processed])

        dense_branch = Dropout(self.heavy_dropout_rate)(dense_branch)
        dense_branch = Dense(384, activation='leaky_relu', kernel_regularizer=self.l1l2)(dense_branch) # Fewer Dense layers, moderate size
        dense_branch = BatchNormalization()(dense_branch)
        dense_branch = Attention()([dense_branch, dense_branch])

        ### Output Layer
        output = Dense(self.action_size)(dense_branch)

        model = kkeras.Model(
            inputs=[candle_inputs, inv_state],
            outputs=output
        )

        tf.keras.utils.plot_model(model, to_file="model_vis.png", show_shapes=True)

        try:
            tf.keras.utils.plot_model(model, to_file="model_vis.png", show_shapes=True)
        except:
            pass

        model.summary()

        model.compile(loss = tf.keras.losses.MeanAbsoluteError(), optimizer=self.optimizer) # , metrics=[Accuracy()] # loss=self.loss, 

        return model

    def remember(self, state0, state1, action, reward, next_state0, next_state1, done, sorce_num):
        """Adds relevant data to memory
        state0 is candle data
        state1 is the ma and inventory data
        """
        index = self.memory_index % self.memory_length # Use modulo for circular buffer
        if self.memory_count >= self.memory_length: # Buffer is full, about to overwrite an old experience
            if index in self.set_a_indicies:
                self.set_a_indicies.remove(index)
            if index in self.set_b_indicies:
                self.set_b_indicies.remove(index)

        self.memory_states0[index] = state0
        self.memory_states1[index] = state1
        self.memory_actions[index] = action
        self.memory_rewards[index] = reward
        self.memory_next_states0[index] = next_state0
        self.memory_next_states1_noact[index] = next_state1[0] # next_state1_noact, next_state1_act_win, next_state1_act_loss
        self.memory_next_states1_act_win[index] = next_state1[1]
        self.memory_next_states1_act_loss[index] = next_state1[2]
        self.memory_dones[index] = done
        self.memory_index += 1
        self.memory_count = min(self.memory_count + 1, self.memory_length) # Track actual number of samples

        if sorce_num == 1:
            self.set_a_indicies.append(index)
        elif sorce_num == 2:
            self.set_b_indicies.append(index)
        else:
            raise Exception("INVALID SORCE_NUM FOR INDICIES")

        # self.memory.append((state0, state1, action, reward, next_state0, next_state1, done))

    def act(self, state0, state1, is_eval=False):
        """Take action from given possible set of actions
        """
        if self.return_zero:
            return 0 if random.random() >= 0.05 else random.randrange(self.action_size - 1) + 1

        if is_eval:
            return np.argmax(self.model.predict([state0, state1], verbose = 0))
        
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if not is_eval and self.first_iter:
            self.first_iter = False
            return random.randrange(self.action_size) # make it non-hold first try

        action_probs = self.model.predict([state0, state1], verbose = 0)

        return np.argmax(action_probs[0])

    def _act(self, state0l, state1l, is_eval=False):
        s0l = np.stack(state0l, axis=0).squeeze()
        s1l = np.stack(state1l, axis=0) #.squeeze()
        probs = self.model.predict([s0l, s1l], verbose = 0)
        if is_eval:
            return [np.argmax(i) for i in probs]
        aa = []
        for i in probs:
            aa.append(np.argmax(i) if random.random() >= self.epsilon else random.randrange(self.action_size))
            self.epsilon *= self.epsilon_decay if self.epsilon > self.epsilon_min else 1
        
        return aa

    def _act2(self, state0l, state1l, is_eval=False):
        s0l = np.stack(state0l, axis=0).squeeze()
        s1l = np.stack(state1l, axis=0) # .squeeze()
        probs = self.model.predict([s0l, s1l], verbose = 0)
        if is_eval:
            return probs
        
        raise Exception("NOT IMPLEMENTED FOR NON EVAL")


    def update_target_network_soft(self): # Removed tau as parameter, using self.tau_value
        """
        Performs a soft update of the target network weights.
        Each weight in the target network is updated towards the online network's weight:
        W_target = tau * W_online + (1 - tau) * W_target
        """
        tau = self.tau_value # Get tau from self.tau_value
        target_weights = self.target_model.get_weights() # Get weights of target network
        online_weights = self.model.get_weights()     # Get weights of online network
        updated_target_weights = [] # To store updated target weights

        for i in range(len(target_weights)): # Iterate through layers/weight matrices
            # Apply soft update formula to each weight matrix/array
            updated_weight = tau * np.array(online_weights[i]) + (1 - tau) * np.array(target_weights[i])
            updated_target_weights.append(updated_weight)

        self.target_model.set_weights(updated_target_weights) # Set the updated weights back to target model

    def train_experience_replay_gpu(self, batch_size, epochs=1):
        """Train on previous experiences in memory using manual gradient application and tf.data.Dataset."""
        if self.memory_index < batch_size:  # Ensure enough experiences for a batch
            return 0.0

        # 1. Efficiently sample random *indices* from the memory using NumPy
        indices = np.random.choice(self.memory_count, batch_size, replace=False)

        # 2. Directly extract batches using NumPy indexing
        batch_states0         = self.memory_states0[indices]
        batch_states1         = self.memory_states1[indices]
        batch_actions         = self.memory_actions[indices]
        batch_rewards         = self.memory_rewards[indices]
        batch_next_states0    = self.memory_next_states0[indices]
        batch_next_states1_noact = self.memory_next_states1_noact[indices]
        batch_next_states1_act_win  = self.memory_next_states1_act_win[indices]
        batch_next_states1_act_loss  = self.memory_next_states1_act_loss[indices]
        batch_dones           = self.memory_dones[indices]

        batch_next_states1 = np.array([
            batch_next_states1_noact[i] if batch_actions[i] == 0 else batch_next_states1_act_win[i] if batch_rewards[i] >= 0 else batch_next_states1_act_loss[i]
            for i in range(batch_size)
        ])


        # 3. Create tf.data.Dataset pipeline
        rl_dataset = create_rl_dataset_pipeline(
            states0=batch_states0,
            states1=batch_states1,
            actions=batch_actions,
            rewards=batch_rewards,
            next_states0=batch_next_states0,
            next_states1=batch_next_states1,
            dones=batch_dones,
            batch_size=batch_size,
            buffer_size=None, # No need to shuffle again here, already shuffled by replay buffer sampling
            normalize_states=False # Set to True if you want to enable state normalization
            # state0_mean=self.state0_mean, # Pass pre-calculated means and stds if normalizing
            # state0_std=self.state0_std,
            # state1_mean=self.state1_mean,
            # state1_std=self.state1_std
        )

        loss_values = [] # To accumulate loss values across batches in this dataset

        # 4. Iterate through the dataset and perform gradient updates
        for (batch_state0, batch_state1, batch_action, batch_reward, batch_next_state0, batch_next_state1, batch_done) in rl_dataset: # Iterate through dataset

            # 5. Predict Q-values for next states (Target model - no gradients here)
            next_q_values = self.target_model.predict([batch_next_state0, batch_next_state1], verbose=0) # Pass batches of tensors
            max_next_q_values = np.max(next_q_values, axis=1)

            # 6. Calculate target Q-values using the Bellman equation
            targets = batch_reward + self.gamma * max_next_q_values # * (1 - batch_done)

            # 7. Update target Q-values for taken actions
            targets_full = tf.zeros_like(next_q_values) # Initialize with zeros or a small value
            targets_full = self.model([batch_state0, batch_state1]) # Get current predictions
            targets_full /= (1 + self.gamma) # Divide all targets to reduce magnitude (reconsider if needed)
            target_indices = tf.stack([tf.range(batch_size), tf.cast(batch_action, dtype=tf.int32)], axis=1) # Create indices for scatter_nd
            updates = tf.reshape(targets, (-1,)) # Reshape targets to be 1D for scatter_nd
            shape = tf.shape(targets_full) # Get shape of targets_full
            targets_full = tf.tensor_scatter_nd_update(targets_full, target_indices, updates) # Use tensor_scatter_nd_update

            clipping_threshold = 5
            targets_full = np.clip(targets_full, -clipping_threshold, clipping_threshold)


            # 8. Manual Gradient Application using tf.GradientTape()
            with tf.GradientTape() as tape:
                # a. Predict Q-values for current states (Current model)
                current_q_values = self.model([batch_state0, batch_state1]) # Pass batches of tensors

                # b. Calculate Loss (Mean Squared Error - MSE)
                loss_value = self.loss(targets_full, current_q_values)

            # 9. Get gradients and apply
            gradients = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            loss_values.append(loss_value.numpy()) # Accumulate loss values

        self.update_target_network_soft()

        return np.mean(loss_values) # Return average loss over batches in the dataset

    def train_experience_replay_2(self, batch_size, epochs = 1):
        """Train on previous experiences in memory
        Looks like it is just a smaller version of the main training model, most likely won't
            need to change this, but not 100% sure. 
        """
        if self.n_iter % self.reset_every == 0:
            self.update_target_network_soft()
        self.n_iter += 1

        if self.memory_index <= batch_size:  # Ensure enough experiences for a batch
            return

        # 1. Efficiently sample random *indices* from the memory using NumPy
        # indices = np.random.choice(self.memory_count, batch_size, replace=False)
        indices = np.array(random.sample(self.set_a_indicies, int(batch_size/2)) + 
                       random.sample(self.set_b_indicies, int(batch_size/2)))

        # 2. Directly extract batches using NumPy indexing (vectorized and very fast!)
        batch_states0      = self.memory_states0[indices]
        batch_states1      = self.memory_states1[indices]
        batch_actions      = self.memory_actions[indices]
        batch_rewards      = self.memory_rewards[indices]
        batch_next_states0 = self.memory_next_states0[indices]
        batch_next_states1_noact = self.memory_next_states1_noact[indices]
        batch_next_states1_act_w = self.memory_next_states1_act_win[indices]
        batch_next_states1_act_l = self.memory_next_states1_act_loss[indices]
        batch_dones        = self.memory_dones[indices]

        batch_next_states1 = np.array([
            batch_next_states1_noact[i] if batch_actions[i] == 0 else batch_next_states1_act_w[i] if batch_rewards[i] > 0 else batch_next_states1_act_l[i]
            for i in range(batch_size)
        ])

        states0      = np.array(batch_states0).squeeze()
        states1      = np.array(batch_states1) # .squeeze()
        next_states0 = np.array(batch_next_states0).squeeze()
        next_states1 = np.array(batch_next_states1) # .squeeze()
        dones        = np.array(batch_dones       )
        actions      = np.array(batch_actions     )
        rewards      = np.array(batch_rewards     )

        # 3. Predict Q-values for current states and next states
        current_q_values = self.model.predict([states0, states1], batch_size=batch_size, verbose=0)
        next_q_values = self.target_model.predict([next_states0, next_states1], batch_size=batch_size, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1) # Max over actions for next states

        # 4. Calculate target Q-values using the Bellman equation
        targets = rewards + self.gamma * max_next_q_values - current_q_values[np.arange(batch_size), actions]
        # targets /= 1 + self.gamma
        ### TRY THIS ###
        # a = current_q_values[np.arange(batch_size), actions]
        ### TRY THIS ###
        # right here, the reward should be reward + self.gamma * max_next_q_values - a

        # 5. Update target Q-values for the taken actions in the current Q-values array
        targets_full = current_q_values.copy()
        targets_full[np.arange(batch_size), actions] = targets # Corrected: use batch_size here

        # clipping_threshold = 5
        # targets_full = np.clip(targets_full, -clipping_threshold, clipping_threshold)

        # f = open("targets.txt", "a")
        # for i, j, k, u in zip(current_q_values, targets_full, actions, rewards):
        #     print("ACTION", k, file = f)
        #     print("Targets", j, file = f)
        #     print("Reward", u, file = f)
        #     # print("ACTION", k, file = f)
        #     # print("ACTION", k, file = f)
        # if self.reset_every % self.n_iter == 0:
        #     print("RESETTING TARGET", file = f)
        # print("Train Again", file = f)
        # f.close()
        # step_decay_callback = LearningRateScheduler(self.learning_rate_sched)

        # 6. Train the current Q-network with states and target Q-values
        lossd = self.model.fit(
            [states0, states1], np.array(targets_full),
            epochs=epochs, verbose=0 # , callbacks = [step_decay_callback]
        )

        # self.update_target_network_soft()

        # print(lossd.history['loss'][0])

        return lossd.history['loss'][0]
    
    def train_experience_replay(self, batch_size, epochs = 1):
        """Train on previous experiences in memory
        Looks like it is just a smaller version of the main training model, most likely won't
            need to change this, but not 100% sure. 
        """
        # mini_batch = random.sample(self.memory, batch_size)

        # DQN
        if self.strategy == "dqn":
            raise "NOT IMPLEMENTED"
            for state0, state1, action, reward, next_state0, next_state1, done in self.memory:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    if reward == 0:
                        target = sigmoid(reward) + self.gamma * np.amax(self.target_model.predict([next_state0, next_state1])[0])
                    else:
                        target = sigmoid(reward) * 2 + self.gamma * np.amax(self.target_model.predict([next_state0, next_state1])[0])

                # estimate q-values based on current state
                q_values = self.model.predict([state0, state1])
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                self.X_train_cd.append(state0[0])
                self.X_train_mav.append(state1[0])
                self.y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            s0l = []
            s1l = []
            # print(self.memory)
            for state0, state1, action, reward, next_state0, next_state1, done in list(self.memory):
                self.X_train_cd.append(state0)
                self.X_train_mav.append(state1)
                # self.y_train.append(rd)

            # q_values = self.target_model.predict([s0l, s1l], verbose = 0)

        # Double DQN
        elif self.strategy == "double-dqn":
            raise "NOT IMPLEMENTED"
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state0, state1, action, reward, next_state, done in self.memory:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict([next_state0, next_state1])[0][np.argmax(self.model.predict(next_state)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict([state0, state1])
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                self.X_train_cd.append(state0[0])
                self.X_train_mav.append(state1[0])
                self.y_train.append(q_values[0])
                
        else:
            raise NotImplementedError()
        
        self.memory = None
        self.memory = deque(maxlen=self.memory_length)


        xr2 = np.stack(self.X_train_cd, axis=0).squeeze()
        xr22 = np.stack(self.X_train_mav, axis=0).squeeze()

        # xr2 = np.array(self.X_train_cd)
        # # x_reshaped = xr2.reshape(xr2.shape[0], xr2.shape[2], xr2.shape[3])
        # xr22 = np.array(self.X_train_mav)
        # # x_reshaped2 = xr22.reshape(xr22.shape[0], xr22.shape[2])
        
        # print(y_train)

        self.n_iter += 1

        # update q-function parameters based on huber loss gradient eval_d

        lossd = self.model.fit(
            [xr2, xr22], np.array(self.y_train),
            epochs=epochs, verbose=1, 
            batch_size=batch_size, validation_split=0.1
        )
        print(np.average(lossd.history['val_loss']))
        print(np.average(lossd.history['loss']))
        return np.average(lossd.history['loss'])
        # accu = lossd.history["accuracy"][0]
    
    def verif_model(self, data):
        X_train, y_train = [], []
        # SELF.WINDOW SIZE NOT SYNCED
        mini_batch = evaluate_model(self, data, self.state_size, False)[5][6]

        # DQN NORMAL NOT UPDATED
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state)
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state)
                y_train.append(q_values[0])

        # Double DQN NOT UPDATED
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state)
                y_train.append(q_values[0])
                
        else:
            raise NotImplementedError()
        
        xr2 = np.array(X_train)
        x_reshaped = xr2.reshape(xr2.shape[0], xr2.shape[2], xr2.shape[3])
        # update q-function parameters based on huber loss gradient
        lossd = self.model.evaluate(
            np.array(x_reshaped), np.array(y_train), verbose=0
        )
        return lossd

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self, mn = None):
        if mn:
            print("LOADING PTM")
            return load_model("models/" + mn, custom_objects=self.custom_objects)
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
