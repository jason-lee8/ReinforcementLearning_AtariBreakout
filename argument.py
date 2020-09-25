"""
Modify arguments using this class.
"""

class Arguments:
    def __init__(self):
        # Run Settings:
        self.train             = True       # True if you want to train the network
        self.load_model_train  = True       # If you want to load a model rather than generating a new one
        self.train_model_path  = 'saved_dqn_networks/breakout_dqn_300000.h5'
        # If you want to save the replay_memory object so when training crashes, you can reload it
        self.save_memory       = True       # Note that this file is extremely large!
        self.load_memory       = False      # If you want to load the saved memory file
        self.load_memory_path  = 'memory/memory_pickle'

        self.test              = False      # If you want to test the network (maxamizes q function)
        self.testEpisodes      = 20         # Number of test games the agent plays
        self.test_model_path   = 'saved_dqn_networks/Breakout_DQN_500000.h5'
        self.render            = False      # If you want to display the game being played

        # Hyperparameters:
        self.gamma             = 0.995      # Discount Factor
        self.initial_epsilon   = 1.0        # Exploration variable start value
        self.final_epsilon     = 0.05       # Exploration variable final value (doesn't go below this value)
        self.exploration_steps = 750000     # Number of exploration steps
        self.learning_rate     = 0.0003     # Rate at which the agent learns Q values at
        self.optimizer         = 'adam'     # Optimizer - either 'adam' or 'rmsprop'
        self.min_grad          = 0.01       # The minimum gradient added into the denomenator of the RMS prop algorithm
        self.batch_size        = 32         # Amount of data that is trained per training

        # Envrionment Info:
        self.env_name              = 'BreakoutNoFrameskip-v4'
        self.frame_width           = 84        # Width of frame
        self.frame_height          = 84        # Height of frame 
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
        self.DDQN    = True         # Double-Deep Q-Network
        self.dueling = False        # Dueling Deep Q-Network

        # Path Info:
        self.save_network_folder = 'saved_dqn_networks/'
        self.save_summary_folder = 'dqn_summary/'
        self.memory_save_path    = 'memory/'

        # Recording:
        self.record    = True       # If you want to record video
        self.video_dir = 'video/'
