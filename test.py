import torch
from gcn import GCN
from graphsage import GraphSage
from utils import *
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt

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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

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

if device != 'cpu':
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)

def test_gnn(adj,gnn):
    ''' test on GCN '''

    adj = normalize_adj_tensor(adj)

    if gnn == 'GCN':
        GNN = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=0.5)

    if gnn == 'GraphSage':
        GNN = GraphSage(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=0.5)

    if device != 'cpu':
        GNN = GNN.to(device)

    optimizer = optim.Adam(GNN.parameters(),
                        lr=args.lr, weight_decay=5e-4)

    GNN.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = GNN(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

    GNN.eval()
    output = GNN(features, adj)

    acc_test = accuracy(output[idx_test], labels[idx_test])

    return acc_test.item()


def main():

    modified_adj = np.loadtxt('GraD_modified_{}_{}.txt'.format(args.dataset,args.ptb_rate),delimiter=' ')
    modified_adj = torch.FloatTensor(modified_adj).cuda()

    runs = 10
    
    gnns = ['GCN','GraphSage']
    for gnn in gnns:
        clean_acc = []
        attacked_acc = []

        print('=== testing {} on original(clean) graph ==='.format(gnn))
        for i in range(runs):
            print('Running the {}-th time'.format(i))
            clean_acc.append(test_gnn(adj,gnn))

        print('=== testing {} on attacked graph ==='.format(gnn))
        for i in range(runs):
            print('Running the {}-th time'.format(i))
            attacked_acc.append(test_gnn(modified_adj,gnn))

        plt.figure(figsize=(6,6))
        sns.boxplot(x=["Acc. Clean", "Acc. Perturbed"], y=[clean_acc, attacked_acc])

        plt.title("Accuracy before/after perturbing {} edges".format(args.ptb_rate*100))
        plt.savefig("results/results_on_{}_{}_{}.png".format(args.dataset,gnn,args.ptb_rate), dpi=600)
        plt.show()
        plt.clf()

        attacked_acc = np.array(attacked_acc)

        print('mean value: %f' % (attacked_acc.mean()))
        print('std value: %f' % (attacked_acc.std()))
        np.savetxt('results/GraD_{}_{}_{}_mean_{}_std_{}.txt'.format(args.dataset,gnn,args.ptb_rate,attacked_acc.mean(),attacked_acc.std()),attacked_acc)

if __name__ == '__main__':
    main()

