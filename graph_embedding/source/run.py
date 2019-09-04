import argparse
import os
import torch
from torch.utils.data import DataLoader
from model import *
from trainer import *
from data import *



class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("--output_path", required=True, type=str, help="output/bert.model")
    parser.add_argument("--bert_path", type=str, help='path of pretrained bert')
    parser.add_argument('--model', type=int, default=0)

    parser.add_argument('--vocab_size', type=int, default=120)
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layers", type=int, default=8, help="number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--max_atom", type=int, default=100, help="maximum sequence len")

    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    d = vars(parser.parse_args())
    args = Option(d)

    if args.model==0:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = Data(args.train_dataset, args.max_atom)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = Data(args.test_dataset, args.max_atom) \
            if args.test_dataset is not None else None
    elif args.model==1:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = Data2(args.train_dataset, args.max_atom)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = Data2(args.test_dataset, args.max_atom) \
            if args.test_dataset is not None else None


    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building model")
    gat = GAT(args.hidden, args.attn_heads, args.vocab_size, args.layers)

    if args.model==0:

        model = MolGraph(gat, args.hidden,args.vocab_size)
        print("Creating Trainer")
        trainer = Trainer(args, gat, model, train_dataloader=train_data_loader, test_dataloader=test_data_loader)
    elif args.model==1:

        model = MolGraph2(gat,args.hidden,args.vocab_size)
        print("Creating Trainer")
        trainer = Trainer2(args, gat, model, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    # model = bert_fc(args.bert_path, args.hidden)



    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == '__main__':
    train()
