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

This is useful because if numExplorationSteps is set to a very large number, the alogorithm will learn the Q-values of actions in all sorts of states that take a lot of quality moves to reach. For example, in the second animation in the demo below, the agent is playing in a state that only happens late in the game. 

After every step (or action) taken by the agent, it will save the to-from pair to a deque called `experience_memory`. This object stores the reward, `r`, received after taking some action, `a`, in state, `s`, which resulted in state, `s'`. This will come in handy during experience replay.

In my program, they are the following variables: `state`, `action`, `reward`, `done`, `next_state`. 

`done` is a boolean that is True if the action taken resulted in losing the game (which means that `state` is a terminal state). Otherwise, `done` is False, meaning the agent could still continue playing. 

**Experience replay** is is a temporal difference learning method that uses the previous states, actions, rewards, and next states stored in the memory to learn to maxamize a Q-function. It does this using this equation here:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686">

Source: [Wikipedia](https://en.wikipedia.org/wiki/Q-learning#Algorithm)

Hyperparameters:
- **α (alpha)** - Learning rate
- **γ (gamma)** - Discount factor

We care about the next state because we want to maxamize the total reward. The agent will rather choose an action with a low immediate reward but high long-term reward than an action with a high immediate reward but has a low long-term reward. This is credited due to the hyperparamter, known as the discount factor, **γ**. Gamma is the amount by which the algorithm discounts future rewards, so if you seek high immediate rewards, you would set gamma to be a small number. Vice versa. 

The actual network learns this new Q-value by subtracting the old prediction by the new prediction and squaring it. In other words, DQNs typically use MSE (mean-squared error) loss functions. After the loss is computed, the network performs gradient descent which adjusts weights/biases based on the derivatives of the activation functions to determine how much each weight/bias was at fault. 

After many, many steps, the agent will be learn to maxamize the reward by abusing the law of large numbers. This is because as experience in the memory increases, any stochastisity in the envrionment will become apparent. Thus, the agent will learn the state-transition probabilities (if I take action, a, in state, s, what is the probability it will land me in state, s'). 

However, my computer kept crashing at around 650,000 steps.

## Demo
Here is an animation of the network playing towards the start of the training process:
<p align="center"> 
<img src="pre_training.gif">
</p>

Here is an example of the network playing Atari Breakout:
<p align="center"> 
<img src="example.gif">
</p>

This is my favorite animation. It shows how big of an impact the discount factor has on the algorithm. As you can see, the agent aims to get the ball stuck on top of the blocks. It blows my mind that using this algorithm, the agent actually became smart!
<p allign="center">
<img src="smart_breakout.gif">
</p>

## Requirements to run
Install these python libraries before running.
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
