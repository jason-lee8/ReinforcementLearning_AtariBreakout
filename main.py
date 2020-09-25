from test import test
from Utils.environment import Environment
from argument import Arguments
from agent_dir.agent_dqn import Agent_DQN


def run(args):
    # All frames are preprocessed with atari wrapper.
    if args.train:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=arg.testEpisodes)


if __name__ == '__main__':
    args = Arguments()
    run(args)
