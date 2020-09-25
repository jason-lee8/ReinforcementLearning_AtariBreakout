import os
import random
import numpy as np
import h5py
import tensorflow as tf
from collections import deque
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, ReLU, Multiply, Maximum, Add, merge
from keras.optimizers import RMSprop,Adam
from keras.initializers import he_uniform
import keras.backend as K
from keras.layers import Lambda
import pickle

from keras.backend.tensorflow_backend import set_session

from Utils import ReplayBuffer

class Agent_DQN():
    def __init__(self, env, Arguments):
        # parameters
        self.frame_width            = Arguments.frame_width
        self.frame_height           = Arguments.frame_height
        self.max_steps              = Arguments.max_num_steps
        self.state_length           = Arguments.agent_history_length
        self.gamma                  = Arguments.gamma
        self.exploration_steps      = Arguments.exploration_steps
        self.initial_epsilon        = Arguments.initial_epsilon
        self.final_epsilon          = Arguments.final_epsilon
        self.replay_memory_size     = Arguments.replay_memory_size
        self.replay_start_size      = Arguments.replay_start_size
        self.batch_size             = Arguments.batch_size
        self.target_update_interval = Arguments.target_update_interval
        self.train_interval         = Arguments.train_interval
        self.learning_rate          = Arguments.learning_rate
        self.min_grad               = Arguments.min_grad
        self.save_interval          = Arguments.save_interval
        self.max_num_no_move_steps  = Arguments.max_num_no_move_steps
        self.model_name             = Arguments.model_name

        # Paths
        self.save_network_folder    = Arguments.save_network_folder
        self.save_summary_folder    = Arguments.save_summary_folder
        self.test_model_path        = Arguments.test_model_path
        self.memory_save_path       = Arguments.memory_save_path
     
        # Environment 
        self.env          = env
        self.num_actions  = env.action_space.n
        
        # Epsilon
        self.epsilon      = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.t            = 0

        # Algorithm types
        self.ddqn                   = Arguments.DDQN
        self.dueling                = Arguments.dueling

        if Arguments.optimizer.lower() == 'adam':
            self.opt = Adam(lr=self.learning_rate)
        else:
            self.opt = RMSprop(lr=self.learning_rate, decay=0, rho=0.95, epsilon=self.min_grad)

        # Input that is not used when fowarding for Q-value 
        # or loss calculation on first output of model 
        self.dummy_input = np.zeros((1, self.num_actions))
        self.dummy_batch = np.zeros((self.batch_size, self.num_actions))

        # Initialize variables and checkpoint stuff
        self.Return         = 0.0
        self.total_q_max    = 0.0
        self.duration       = 0
        self.episode        = 0
        self.loss           = 0.0
        self.running_average = deque()
        if not os.path.exists(self.save_network_folder):
            os.makedirs(self.save_network_folder)
        if not os.path.exists(self.save_summary_folder):
            os.makedirs(self.save_summary_folder)

        # Create experience replay memory
        if Arguments.load_memory:
            with open(self.memory_save_path + 'memory_pickle', 'rb') as file:
                self.replay_memory = pickle.load(file)
        else:
            self.replay_memory = deque()

        # Create Q network and target network
        if Arguments.load_model_train:
            self.q_network      = load_model(Arguments.train_model_path)
            self.target_network = load_model(Arguments.train_model_path)
        else:
            self.q_network      = self.build_network()
            self.target_network = self.build_network()

        if Arguments.test:
            self.q_network.load_weights(self.test_model_path)
        else:
            self.log = open(self.save_summary_folder+self.model_name+'.log','w')

        # Set target_network weights
        self.target_network.set_weights(self.q_network.get_weights())


    def train(self):
        while self.t <= self.max_steps:
            done = False
            observation = self.env.reset()
            for _ in range(random.randint(1, self.max_num_no_move_steps)):
                prev_observation = observation
                observation, _, _, _ = self.env.step(0)  # Do nothing
            while not done:
                prev_observation = observation
                action = self.training_action(prev_observation)
                observation, reward, done, _ = self.env.step(action)

                next_state = observation

                # Store transition in replay memory
                self.replay_memory.append((prev_observation, action, reward, next_state, done))
                if len(self.replay_memory) > self.replay_memory_size:
                    self.replay_memory.popleft()

                if self.t >= self.replay_start_size:
                    # Train network every {train_interval} steps
                    if self.t % self.train_interval == 0:
                        self.experience_replay()

                    # Update target network
                    if self.t % self.target_update_interval == 0:
                        self.target_network.set_weights(self.q_network.get_weights())

                    # Save network and memory
                    if self.t % self.save_interval == 0:
                        save_path = self.save_network_folder + '/' + self.model_name+'_'+str(self.t)+'.h5'
                        self.q_network.save(save_path)
                        
                        with open(self.memory_save_path + 'memory_pickle', 'wb') as file:
                            pickle.dump(self.replay_memory, file)

                        print('Successfully saved: ' + save_path)
                        print('Successfully saved: ' + self.memory_save_path + 'memory_pickle')

                self.Return       += reward
                self.total_q_max  += np.max(self.q_network.predict([np.expand_dims(prev_observation, axis=0), self.dummy_input])[0])
                self.duration     += 1

                if done:
                    # Average last 20 rewards
                    self.running_average.append(self.Return)
                    if len(self.running_average) > 20:
                        self.running_average.popleft()

                    # Prints in command prompt
                    print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / REWARD: {4:3.1f} / AVG_REWARD: {5:5.3f} / AVG_MAX_Q: {6:2.4f} / AVG_LOSS: {6:5.5f}'.format(
                        self.episode + 1, self.t, self.duration, self.epsilon, self.Return,
                        np.mean(self.running_average),            # Avg Reward
                        self.total_q_max / float(self.duration), # Avg Max Q
                        self.loss))
                    # Prints in log file
                    print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / REWARD: {4:3.1f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:5.5f}'.format(
                        self.episode + 1, self.t, self.duration, self.epsilon,
                        np.mean(self.running_average), 
                        self.total_q_max / float(self.duration),
                        self.loss), file=self.log)

                    self.Return = 0
                    self.total_q_max = 0
                    self.duration = 0
                    self.episode += 1

                self.t += 1


    def test_action(self, observation):
        """
        Add a random action to avoid the model getting stuck under certain situations
        Input:
            observation: np.array
                A stack of the 4 last preprocessed frames - shape: (84, 84, 4)
        Return:
            action: int
                Action that maxamizes Q
        """
        if 0.05 > random.random():
            return random.randrange(self.num_actions)
        else:
            return np.argmax(self.q_network.predict([np.expand_dims(observation,axis=0), self.dummy_input])[0])


    def training_action(self, observation):
        """
        Add random action to avoid the model getting stuck under certain situations
        Input:
            observation: np.array
                A stack of the 4 last preprocessed frames - shape: (84, 84, 4)
        Return:
            action: int
                Either a random action or an action that maxamizes Q
        """
        if self.epsilon >= random.random() or self.t < self.replay_start_size:
           action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_network.predict([np.expand_dims(observation,axis=0), self.dummy_input])[0])
        # Anneal epsilon linearly over time
        if self.epsilon > self.final_epsilon and self.t >= self.replay_start_size:
            self.epsilon -= self.epsilon_step
        return action


    def experience_replay(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            done_batch.append(data[4])

        # Multiply by 1 so that True --> 1 and False --> 0
        done_batch = np.array(done_batch) + 0
        # Q value from target network
        target_q_values_batch = self.target_network.predict([list2np(next_state_batch), self.dummy_batch])[0]

        if self.ddqn:
            next_action_batch = np.argmax(self.q_network.predict([list2np(next_state_batch), self.dummy_batch])[0], axis=-1)
            for i in range(self.batch_size):
                # (1 - done_batch[i]) because if batch is done --> 1-1 = 0 and there will be no future reward
                y_batch.append(reward_batch[i] + 
                    (1 - done_batch[i]) * self.gamma * target_q_values_batch[i][next_action_batch[i]] )
            y_batch = list2np(y_batch)
        else:
            y_batch = reward_batch + (1 - done_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)
        
        # Create a one hot vector for all batch and num actions
        a_one_hot = np.zeros((self.batch_size, self.num_actions))
        for idx,ac in enumerate(action_batch):
            a_one_hot[idx, ac] = 1.0

        # Train batch and calculate loss
        self.loss = self.q_network.train_on_batch([list2np(state_batch), a_one_hot], [self.dummy_batch, y_batch])
        self.loss = self.loss[1]


    def build_network(self):
        """
        Creates the Atari network that DeepMind used in their paper - "Human-level Control Through Deep Reinforcement Learning"
        Network overview:

            Input: (84, 84, 4)
            32 filters of 8 x 8 with stride 4 followed by relu
            64 filters of 4 x 4 with stride 2 followed by relu
            64 filters of 3 x 3 with stride 1 followed by relu
            Fully connected layer w/ 512 nodes with relu activation
            Fully connected w/ [action_space] nodes
        """
        # Consturct model
        
        input_frame    = Input(shape=(self.frame_width, self.frame_height, self.state_length))
        action_one_hot = Input(shape=(self.num_actions, ))
        
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer=he_uniform())(input_frame)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=he_uniform())(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=he_uniform())(conv2)
        flat_feature       = Flatten()(conv3)
        hidden_feature     = Dense(512, activation='relu', kernel_initializer=he_uniform())(flat_feature)
        q_value_prediction = Dense(self.num_actions)(hidden_feature)

        if self.dueling:
            # Dueling Network
            # Q = Value of state + (Value of Action - Mean of all action value)
            hidden_feature_2 = Dense(512,activation='relu', kernel_initializer=he_uniform())(flat_feature)
            state_value_prediction = Dense(1)(hidden_feature_2)
            q_value_prediction =  Lambda(lambda x: x[0] - K.mean(x[0]) + x[1], 
                                         output_shape=(self.num_actions,))([q_value_prediction, state_value_prediction])

        select_q_value_of_action = Multiply()([q_value_prediction, action_one_hot])
        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(select_q_value_of_action)
        
        model = Model(inputs=[input_frame, action_one_hot], outputs=[q_value_prediction, target_q_value])
        
        # MSE loss only on state value prediction
        model.compile(loss=['mse','mse'], loss_weights=[0.0, 1.0], optimizer=self.opt)

        return model


def list2np(input_list):
    return np.float32(np.array(input_list))


def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)