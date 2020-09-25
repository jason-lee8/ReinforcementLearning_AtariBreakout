# Deep-Q-Network--Breakout
I used Keras to implement a Deep-Q-Network that can play Atari Breakout at an above human-level performance.

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
