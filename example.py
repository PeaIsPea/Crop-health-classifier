from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Cnn training")
    parser.add_argument('--root', '-r', type=str, default="./data", help='Root of dataset')
    parser.add_argument('--epochs','-e', type=int, default=100, help='Number of epochs')

    parser.add_argument('--batch_size','-b', type=int, default=8, help='Number of batch_size')
    parser.add_argument('--images_size','-i', type=int, default=112, help='Size of Image')
    parser.add_argument('--logging', '-l', type=str, default="./board")
    parser.add_argument('--trained_models', '-m', type=str, default="trained_models")
    parser.add_argument('--checkpoints', '-c', type=str, default=None)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    print("number of epochs {}" .format(args.epochs))
    print("number of batch size {}".format(args.batch_size))
