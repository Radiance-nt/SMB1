import argparse
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from common.agent import SPRAgent
from common.env import *

import tianshou as ts
import torch
from tianshou.data import VectorReplayBuffer

from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger

from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='Trains Super Mario Bros 1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--exp_name', type=str, default='ppo_spr', help='The name of this experiment')
    parser.add_argument('--world', type=int, default=1, help='Select World')
    parser.add_argument('--stage', type=int, default=1, help='Select Stage')
    # Optimization options
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--spr_batch_size', type=int, default=1024)
    parser.add_argument('--spr_lr', type=float, default=0.0001)

    parser.add_argument('--collector_env_num', type=int, default=10)
    parser.add_argument('--evaluator_env_num', type=int, default=5)
    parser.add_argument('--n_evaluator_episode', type=int, default=1)

    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--step_per_collect', type=int, default=500)
    parser.add_argument('--update_per_collect', type=int, default=10)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=10000)

    parser.add_argument("--spr", default=False,  action='store_true', help='Use Spr or not')
    parser.add_argument("--noise", type=float, default='1.75', help='GaussianNoise noise rate')

    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log', type=str, default='./log/', help='Log folder.')

    args = parser.parse_args()
    return args


def main(args):
    default_log = './log/{}-{}/{}'.format('1', '1', time.strftime("%b_%d_%H_%M", time.localtime()))
    log_path = default_log if args.log == './log/' else './log/{}/{}-{}/'.format(
        args.log, args.world, args.stage)
    tb_logger = SummaryWriter(log_path)
    save_path = os.path.join(log_path, 'data')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    def get_envs(num):
        def build_env(world=1, stage=1, output_path=None):
            env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(args.world, args.stage))
            if output_path:
                monitor = Monitor(256, 240, output_path)
            else:
                monitor = None

            env = JoypadSpace(
                env,
                [['right'],
                 ['right', 'A']]
            )
            env = CustomReward(env, world, stage, monitor)
            env = SkipFrame(env, skip=4)
            env = GrayScaleObservation(env, keep_dim=False)
            env = ResizeObservation(env, shape=84)
            env = TransformObservation(env, f=lambda x: x / 255.)
            env = FrameStack(env, num_stack=4)
            return env

        if num == 0:
            return None
        envs = ts.env.DummyVectorEnv([build_env for _ in range(num)])
        return envs

    train_envs = get_envs(args.collector_env_num)
    test_envs = get_envs(args.evaluator_env_num)

    policy = SPRAgent((4, 84, 84), train_envs.action_space[0].n, 256, args)

    train_collector = ts.data.Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, args.collector_env_num),
        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs,
                                       exploration_noise=True) if test_envs is not None else None

    def save_fn(policy, info='best'):
        torch.save(policy.state_dict(), os.path.join(save_path, 'policy_{}.pth'.format(info)))

    result = onpolicy_trainer(
        policy, train_collector, test_collector, max_epoch=args.max_epoch,
        step_per_epoch=args.step_per_epoch, step_per_collect=args.step_per_collect,
        repeat_per_collect=args.update_per_collect,
        episode_per_test=args.n_evaluator_episode,
        batch_size=args.batch_size, logger=TensorboardLogger(tb_logger),
        save_fn=save_fn,
        stop_fn=None)

    print(f'Finished training! Use {result["duration"]}')


if __name__ == "__main__":
    args = get_args()
    main(args)
