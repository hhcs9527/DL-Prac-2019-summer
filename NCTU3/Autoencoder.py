import argparse
import torch.nn as nn
from Trainer import Autoencoder


def main(args):
    criterion = nn.CrossEntropyLoss()

    auto = Autoencoder(args.hidden_size, args.cond_embed_size, 
                        args.output_size, args.target_path, criterion, args.epoch, 
                        args.train_or_not, args.lr, args.input_embed_size, args.teacher_forcing_ratio, args.ratio_kind)
    if Train:
        if args.ratio_kind <= 3:
            auto.writefile()
        else:
            auto.train()
    else:
        auto.test()
        auto.generate()


if __name__ == "__main__":
    Train = True
    Train = False
    parser = argparse.ArgumentParser()
    # -- reprsent the var, no - -> you must input something, if try to specific refer use -o/-oo(define my self)
    # directly read, no need typing
    # parser.add_argument('--target_path', default = './lab3/train.txt', type=str) 
    #
    # must type
    # parser.add_argument('lr', default = 0.001, type=float)
    #
    # type -oo then the variable
    # parser.add_argument('-oo', '--epoch',default = 10, type=int)
    if Train:
        print('Training')
        parser.add_argument('--target_path', default = './lab3/train.txt', type=str)
        parser.add_argument('--train_or_not', default = True, type=int)
    else:
        print('Testing')
        parser.add_argument('--target_path', default = './lab3/test_data.txt', type=str)
        parser.add_argument('--train_or_not', default = False, type=int)

    parser.add_argument('--lr', default = 0.001, type=float)
    parser.add_argument('--epoch',default = 300, type=int)
    parser.add_argument('--hidden_size', default = 256, type=int)
    parser.add_argument('--cond_embed_size', default = 8, type=int)
    parser.add_argument('--output_size', default = 28, type=int)
    parser.add_argument('--input_embed_size', default = 64, type=int)
    parser.add_argument('--teacher_forcing_ratio', default = 1, type=int)
    parser.add_argument('--ratio_kind', default = 3, type = int)
    args = parser.parse_args()
    main(args)