import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=15140)
    parser.add_argument('--visual_dim', type=int, default=2048)
    parser.add_argument('--user_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=64)

    args = parser.parse_args()
    return args
