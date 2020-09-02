import matplotlib
matplotlib.use('Agg')

import torch
import sys
import os.path
sys.path.append('%s/../DenoisingGCN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
import networkx as nx
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csr_matrix, csc_matrix
from sklearn import metrics
import scipy.stats as ss
import torch.nn.functional as F
import cPickle as pickle

parser = argparse.ArgumentParser(description='E-Net')
# general settings
parser.add_argument('--data-name', default='citeseer', help='network name')
parser.add_argument('--save-name', default='0', help='save model name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--missing-ratio', type=float, default=0.1,
                    help='ratio of missing links')
parser.add_argument('--split-ratio',type=str,default='0.8,0.1,0.1',
                    help='ratio of train, val and test links')
parser.add_argument('--neg-pos-ratio', type=float, default=5,
                    help='ratio of negative/positive links')
# model settings
parser.add_argument('--hop', default=2, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=20,
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--embedding-size', default=128,
                    help='embedding size of node2vec')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
parser.add_argument('--lazy-subgraph', action='store_true', default=True,
                    help='whether to use lazy subgraph extraction')
parser.add_argument('--multi-subgraph', default=3,
                    help='number of subgraphs to extract for each queried nodes')
parser.add_argument('--num-walks', default=5,
                    help='number of walks for each node')
parser.add_argument('--num-node-to-walks', default=5,
                    help='number of walks for each node')
parser.add_argument('--trainable-noise', action='store_true', default=False,
                    help='whether to use logistic regression on noise detection')
parser.add_argument('--mp', action='store_true', default=False, help='whether to use multi processing')
# earlystopping setting
parser.add_argument('--early-stop', default=True, help='whether to use early stopping')
parser.add_argument('--early-stop-patience', type=int, default=7, help='patience for early stop')
parser.add_argument('--early-stop-index', type=int, default=0, help='early stop index')
parser.add_argument('--early-stop-delta', type=float, default=1e-3, help='early stop delta')
# GCN setting
parser.add_argument('--learning-rate', type=float, default=1e-4, help='GCN learning rate')
parser.add_argument('--smooth-coef', type=float, default=1e-4, help='smooth regularization coefficient')
parser.add_argument('--noise-hidden-dim', type=int, default=300, help='noise detector hidden dimension')
parser.add_argument('--reg-smooth', action='store_true', default=False,
                    help='whether to use smooth regularization')
parser.add_argument('--pr-threshold', type=float, default=0, help='noise threshold')
# Noise Detector setting
parser.add_argument('--use-sig', action='store_true', default=False, help='whether to use sigmoid')
parser.add_argument('--use-soft', action='store_true', default=False, help='whether to use softmax')
parser.add_argument('--noise-bias', action='store_true', default=False, help='whether to use noise bias')
parser.add_argument('--noise-init', action='store_true', default=False, help='whether to init noise weights')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
torch.cuda.manual_seed_all(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
torch.backends.cudnn.deterministic = True

if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)
if args.num_walks is not None:
    args.num_walks = int(args.num_walks)
if args.num_node_to_walks is not None:
    args.num_node_to_walks = int(args.num_node_to_walks)
args.embedding_size = int(args.embedding_size)
args.split_ratio = [float(_) for _ in args.split_ratio.split(',')]

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
print("input: {}".format(args.data_dir))
data = sio.loadmat(args.data_dir)
label = data['label']
net_c = data['net']   # input: clean network (not consider missing links)

if args.data_name == 'cora' or args.data_name == 'citeseer' or args.data_name == 'pubmed':
    data = sio.loadmat(os.path.join(args.file_dir, 'data/{}_n0.1.mat'.format(args.data_name)))
else:
    data = sio.loadmat(os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name)))
net = data['net']   # input: flawed network
node_num = net.shape[0]

# edges; clean edges; noisy edges
edges = nx.from_scipy_sparse_matrix(net).edges()
edges_clean = nx.from_scipy_sparse_matrix(net_c).edges()
print('# of nodes: {}'.format(node_num))
print('# of clean/all edges: {}/{}'.format(len(edges_clean), len(edges)))

num_noisy_edge = len(edges) - len(edges_clean)

if data.has_key('group'):
    # load node attributes (here a.k.a. node classes)
    try:
        attributes = data['group'].toarray().astype('float32')
    except AttributeError:
        attributes = data['group'].astype('float32')
else:
    attributes = None
print('attribute dimension: {}'.format(attributes.shape))
# check whether net is symmetric (for small nets only)
if False:
    net_ = net.toarray()
    assert(np.allclose(net_, net_.T, atol=1e-8))
#Sample train and test links
train_val_test, missing_links = sample_train_val_test(edges, edges_clean, net, args.split_ratio, args.missing_ratio,
                                       args.neg_pos_ratio, max_train_num=args.max_train_num)
print('MISSING LINK PREDICTION-- # train_pos: %d, # train_neg: %d, # val_pos: %d, # val_neg: %d, # test_pos: %d, # test_neg: %d' % (
    len(train_val_test['train'][0][0]), len(train_val_test['train'][1][0]), len(train_val_test['val'][0][0]),
    len(train_val_test['val'][1][0]), len(train_val_test['test'][0][0]), len(train_val_test['test'][1][0])))

'''Train and apply classifier'''
A = net.copy()  # the observed network
# mask missing links
for key, value in train_val_test.items():
    A[value[0][0], value[0][1]] = 0
    A[value[0][1], value[0][0]] = 0

# construct noisy link candidate
pos_noisy_links = [e for e in edges if e not in edges_clean and e[::-1] not in edges_clean]
remain_clean_links = [e for e in edges_clean if e not in missing_links and e[::-1] not in missing_links]
perm = random.sample(range(len(remain_clean_links)), len(pos_noisy_links) * args.neg_pos_ratio)
neg_noisy_links = [remain_clean_links[i] for i in perm]
noisy_candidate = pos_noisy_links + neg_noisy_links
row = np.array(noisy_candidate)[:,0]; col = np.array(noisy_candidate)[:,1]
data = [1 for i in range(len(row))]
Identity = csc_matrix((data, (row,col)), shape=A.shape)

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, args.embedding_size, True, train_val_test['train'][1])
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

''' Construct data for noisy link detection '''
edge_information = np.loadtxt('data/{}.score'.format(args.data_name))
edge_information = {(min(x, y), max(x, y)): edge_information[i] for i, (x, y) in enumerate(edges)}

graph_fn = 'graph/{}_{}_{}_{}.pkl'.format(args.data_name, args.hop, args.max_nodes_per_hop, args.lazy_subgraph)
if os.path.exists(graph_fn):
    print('load subgraphs')
    graphs, max_n_label = pickle.load(open(graph_fn, 'r'))
    print('finish loading subgraphs')
else:
    graphs, max_n_label = links2subgraphs(A, Identity, train_val_test, args.hop,
                                          args.max_nodes_per_hop, node_information, edge_information,
                                          args.lazy_subgraph, args.multi_subgraph,
                                          args.num_node_to_walks, args.num_walks, args.mp)
    # pickle.dump([graphs, max_n_label], open(graph_fn, 'w'))

train_graphs, val_graphs, test_graphs = graphs['train'], graphs['val'], graphs['test']
print('# train graph: %d, #val graph: %d, # test graph: %d' % (len(train_graphs), len(val_graphs), len(test_graphs)))

# Enet configurations
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32,32,32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = 100
cmd_args.learning_rate = args.learning_rate
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
cmd_args.decay_learning_rate = False
cmd_args.trainable_noise = args.trainable_noise

cmd_args.noise_matrix = True
cmd_args.reg_smooth = args.reg_smooth
cmd_args.loss_missing = True
cmd_args.nodefeat_lp = True

cmd_args.smooth_coef = args.smooth_coef
cmd_args.pr_threshold = args.pr_threshold
cmd_args.total_num_nodes = net.shape[0]

cmd_args.use_sig = args.use_sig
cmd_args.use_soft = args.use_soft
cmd_args.noise_bias = args.noise_bias
cmd_args.noise_init = args.noise_init
print(cmd_args)

if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs + val_graphs])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))
suffix = '{}.nw-{}.l-{}.s-{}.nd-{}.lr-{}'.format(args.data_name, args.num_walks, args.num_node_to_walks,
                                                 args.smooth_coef, args.noise_hidden_dim, args.learning_rate)


class EarlyStop(object):
    def __init__(self, patience=args.early_stop_patience,
                 index=args.early_stop_index, delta=args.early_stop_delta):
        self.patience = patience
        self.delta = delta
        self.index = index
        self.best_loss = 1e15
        self.test_loss = 1e15
        self.wait = 0
        self.finish = 1

    def check(self, val_loss, test_loss, epoch):
        if val_loss[self.index] - self.best_loss < - self.delta:
            self.best_loss = val_loss[self.index]
            self.test_loss = test_loss
            self.epoch = epoch
            self.wait = 1
        else:
            self.wait += 1
        return self.wait > self.patience


classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_graphs)))
early_stop = EarlyStop()

for epoch in range(cmd_args.num_epochs):
    t0 = time.time()

    random.shuffle(train_idxes)
    avg_loss, _ = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
          'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
          (epoch, avg_loss[0], avg_loss[1], avg_loss[-1], avg_loss[2], avg_loss[3], avg_loss[4],
           avg_loss[5], avg_loss[6], time.time() - t0))

    classifier.eval()
    val_loss, _ = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))), test=True)

    test_loss, _ = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))), test=True)
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
          'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
          (epoch, test_loss[0], test_loss[1], test_loss[-1], test_loss[2], test_loss[3], test_loss[4],
           test_loss[5], test_loss[6], time.time() - t0))

    if early_stop.check(val_loss, test_loss, epoch):
        print("------ early stopping ------")
        test_loss = early_stop.test_loss
        print('\033[93mearlystop missing test of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
              'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
              (epoch, test_loss[0], test_loss[1], test_loss[-1], test_loss[2], test_loss[3], test_loss[4],
               test_loss[5], test_loss[6], time.time() - t0))
        break

if args.reg_smooth and args.lazy_subgraph:
    torch.save(classifier, 'model/{}.pkl'.format(suffix+args.save_name))
