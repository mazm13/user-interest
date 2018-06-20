import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # fixed super-parameters
    parser.add_argument('--num_users', type=int, default=15140)
    parser.add_argument('--num_attributes', type=int, default=80)
    parser.add_argument('--num_ages', type=int, default=40)

    # model super-parameters
    parser.add_argument('--face_k', type=int, default=32)
    parser.add_argument('--visual_dim', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--user_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)

    # validation super-parameters
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--valid_shuffle', type=bool, default=True)

    # train super-parameters
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_epoches', type=int, default=10)

    # misc options
    parser.add_argument('--model_path', type=str, default='models')

    args = parser.parse_args()
    return args
