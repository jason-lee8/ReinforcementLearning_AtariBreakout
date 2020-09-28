# Deep-Q-Network--Breakout
I used Keras to implement a Deep-Q-Network that can play Atari Breakout at an above human-level performance.

## How it Works
The network learns to play video games by using Q-learning, which is a model-free reinforcement learning algorithm. In other words, it uses an algorithm that basically uses trial and error to learn an action-selection policy that will maxamize the expected value of the total reward (also known as the return). 

The way Q-learning works is that the agent has a variable, **ε** (epsilon), that represents the ratio of exploration to exploitation. Exploring means taking a random action while exploiting means taking the action that will maxamize the Q-value of that state. Over time, ε decays to the hyperparameter, minimum ε. 

```python
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = (epsilon - epsilon_min) / numExplorationSteps

if random_number > epsilon: 
    return some random action
`else:
    return action that maxamizes the Q-function

if epsilon > epsilon_min:
    epsilon = epsilon - epsilon_decay
else:
    pass
```

This is useful because if numExplorationSteps is set to a very large number, the alogorithm will learn the Q-values of actions in all sorts of states that take a lot of quality moves to reach. For example, in the demo below, the agent is playing in a state that only happens late in the game. 

After every step (or action) taken by the agent, it will save the to-from pair to a deque called `experience_memory`. This object stores the reward, `r`, received after taking some action, `a`, in state, `s`, which resulted in state, `s'`. This will come in handy during experience replay.

In my program, they are the following variables: `state`, `action`, `reward`, `done`, `next_state`. 

`done` is a boolean that is True if the action taken resulted in losing the game (which means that `state` is a terminal state). Otherwise, `done` is False, meaning the agent could still continue playing. 

**Experience replay** is when the agent uses previous states, actions, rewards, and next states to adjust the weights/biases in its network. It does this using this equation here:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686">

We care about the next state because we want to maxamize the total reward. This means that if some action, `a_1`, has a high immediate reward but puts the agent into a state where it is hard to find any other rewards, then the agent will learn not to choose that action.


Usually, the agent will accumilate 

The agent then uses this to modify its network using the derivatives of the activation functions and biases to more accurately 

It does this by abusing the law of large numbers, which says that an increase in sample size results in more accurate averages. Over many iterations of training on game data saved in the memory, the network is able to 

## Demo
Here is an example of the network playing Atari Breakout:
<p align="center"> 
<img src="example.gif">
</p>

Here is an animation of the network playing towards the start of the training process:
<p align="center"> 
<img src="pre_training.gif">
</p>

## Requirements to run
- gym
- gym[atari]
- tensorflow
- keras
- h5py
- numpy
- pickle (optional)


## How to use
Use the argument.py file to modify the arguments to your liking.

Then, run the main.py file by entering the command, `python main.py` into the terminal.

## References
- [OpenMind Paper - "Human-level control through deep reinforcement learning"](http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
