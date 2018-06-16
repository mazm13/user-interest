import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # model super-parameters
    parser.add_argument('--num_users', type=int, default=15140)
    parser.add_argument('--num_perps', type=int, default=80)
    parser.add_argument('--visual_dim', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--user_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)

    # validation super-parameters
    parser.add_argument('--valid_size', type=float, default=0.05)
    parser.add_argument('--valid_shuffle', type=bool, default=True)

    # train super-parameters
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_epoches', type=int, default=10)

    # misc options
    parser.add_argument('--model_path', type=str, default='models')

    args = parser.parse_args()
    return args
