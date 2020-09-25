
class trainArguments:
    def __init__(self):
        # Run Settings:
        self.train             = True
        self.load_model_train  = True
        self.train_model_path  = 'saved_dqn_networks/breakout_dqn_300000.h5'
        self.load_memory       = True
        self.load_memory_path  = 'memory/memory_pickle'

        self.test              = False
        self.test_model_path   = 'saved_dqn_networks/Breakout_DQN_500000.h5'
        self.render            = False

        # Hyperparameters:
        self.gamma             = 0.995      # Discount Factor
        self.initial_epsilon   = 0.2084     # Exploration variable start value
        self.final_epsilon     = 0.05       # Exploration variable final value (doesn't go below this value)
        self.exploration_steps = 50000      # Number of exploration steps
        self.learning_rate     = 0.0003     # Rate at which the agent learns Q values at
        self.optimizer         = 'adam'     # Optimizer - either 'adam' or 'rmsprop'
        self.min_grad          = 0.01       # The minimum gradient added into the denomenator of the RMS prop algorithm
        self.batch_size        = 32         # Amount of data that is trained per training

        # Envrionment Info:
        self.env_name              = 'BreakoutNoFrameskip-v4'
        self.frame_width           = 84        
        self.frame_height          = 84
        self.agent_history_length  = 4         # Number of most recent frames used in state
        self.max_num_no_move_steps = 4         # Maximum number of 'no move' steps
        
        # Training Info:
        self.model_name             = 'Breakout_DQN'
        self.max_num_steps          = 2000000         # Training ends after this number of steps is taken
        self.replay_start_size      = 0               # Experience replay starts after the replay size > this value
        self.replay_memory_size     = 2000000         # Updates are sampled from this number of most recent frames
        self.save_interval          = 50000           # How often model is saved
        self.target_update_interval = 10000           # How often the target model is updated
        self.train_interval         = 4               

        # Algorithm Type:
        self.DDQN    = True
        self.dueling = False

        # Path Info:
        self.save_network_folder = 'saved_dqn_networks/'
        self.save_summary_folder = 'dqn_summary/'
        self.memory_save_path    = 'memory/'

        # Recording:
        self.record    = True
        self.video_dir = 'video/'


class testArguments:
    def __init__(self):
        # Run Settings:
        self.train             = False
        self.test              = True
        self.test_model_path   = 'old_saved_networks/breakout_dqn_3000000.h5'
        self.render            = True
        self.load_model_train  = False
        self.train_model_path  = 'saved_dqn_networks/breakout_dqn_4000000.h5'

        # Hyperparameters:
        self.gamma             = 0.999      # Discount Factor
        self.initial_epsilon   = 1.0        # Exploration variable start value
        self.final_epsilon     = 0.1        # Exploration variable final value (doesn't go below this value)
        self.exploration_steps = 1000000    # Number of exploration steps
        self.learning_rate     = 0.00025    # The rate at which the agent learns to fix the Q values at
        self.optimizer         = 'rmsprop'  # Optimizer - either 'adam' or 'rmsprop'
        self.min_grad          = 0.01       # The minimum gradient that is added into the denomenator of the RMS prop algorithm
        self.batch_size        = 32         # Amount of data that is trained per training

        # Envrionment Info:
        self.env_name               = 'BreakoutNoFrameskip-v4'
        self.frame_width            = 84        
        self.frame_height           = 84
        self.agent_history_length   = 4         # 
        self.max_num_no_move_steps  = 4         # Maximum number of 'no move' steps
        
        # Training Info:
        self.model_name             = 'Breakout_DQN'
        self.max_num_steps          = 1000000         # Training ends after this number of steps is taken
        self.replay_start_size      = 50000           # Experience replay starts after the replay size > this value
        self.replay_memory_size     = 1000000         # Updates are sampled from this number of most recent frames
        self.save_interval          = 50000           # How often model is saved
        self.target_update_interval = 10000           # How often the target model is updated
        self.train_interval         = 4               

        # Algorithm Type:
        self.DDQN    = False
        self.dueling = False

        # Path Info:
        self.save_network_folder = 'saved_dqn_networks/'
        self.save_summary_folder = 'dqn_summary/'

        # Recording:
        self.record    = True
        self.video_dir = 'video/'
