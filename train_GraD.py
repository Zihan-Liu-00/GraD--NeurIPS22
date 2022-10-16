import torch
from utils import *
import argparse
import numpy as np
from attack import GraD

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same seed also used in other baselines
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

def main():

    # loading dataset
    adj, features, labels = load_data(dataset=args.dataset)
    nclass = max(labels) + 1

    val_size = 0.1
    test_size = 0.8
    train_size = 1 - test_size - val_size

    idx = np.arange(adj.shape[0])
    idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
    idx_unlabeled = np.union1d(idx_val, idx_test)
    perturbations = int(args.ptb_rate * (adj.sum()//2))

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # Generate perturbed graph
    model = GraD(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                        nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                        train_iters=100, attack_features=False, lambda_=0, device=device, args=args)

    if device != 'cpu':
        adj = adj.to(device)
        features = features.to(device)
        labels = labels.to(device)
        model = model.to(device)

    modified_adj = model(features, adj, labels, idx_train,
                        idx_unlabeled, perturbations)
    modified_adj = modified_adj.detach()
    np.savetxt('GraD_modified_{}_{}.txt'.format(args.dataset,args.ptb_rate),modified_adj.cpu().numpy())

if __name__ == '__main__':
    main()

