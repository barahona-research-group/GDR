# This file is part of GDR.
#
# Copyright (C) 2019, Alexis Arnaudon (alexis.arnaudon@imperial.ac.uk), 
# Robert Peach (r.peach13@imperial.ac.uk)
# https://github.com/barahona-research-group/GDR
#
# GDR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GDR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GDR.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import scipy.sparse as sp
import sklearn.metrics.pairwise as sk
from sklearn.metrics import accuracy_score
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def accuracy_predictions(class_H, data_type, weights=None):
    """
    Calculates the accuracy of classifications
    """
   
    # Finding index of columns with known classes to compare
    indices = np.nonzero(data_type)[0]
    
    y_real = np.nonzero(data_type)[1]
    
    y_pred = np.nonzero(class_H)[1][indices]
    y_pred = y_pred % data_type.shape[1]

    if weights is not None:
        weights = np.nonzero(class_H)[1][indices]
        weights = weights % data_type.shape[1]

    accuracy = accuracy_score(y_real, y_pred, sample_weight = weights)

    return (y_pred, y_real, accuracy)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)



def define_MLP_distribution(dataset, directed = False):
	from GCN import MLP
	import tensorflow as tf
	import time
	
	# Set random seed
	seed = 123
	np.random.seed(seed)
	tf.set_random_seed(seed)
	
	def del_all_flags(FLAGS):
	    flags_dict = FLAGS._flags()    
	    keys_list = [keys for keys in flags_dict]    
	    for keys in keys_list:
	        FLAGS.__delattr__(keys)
	
	del_all_flags(tf.flags.FLAGS)

	# Settings
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
	flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
	flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
	flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
	flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
	flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
	flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

	
	num_supports = 1
	if directed:	
		adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_directed_data()	
	else:	
		adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)		
	model_func = MLP
	features = preprocess_features(features)		
	support = [preprocess_adj(adj)]
		
	placeholders = {
	    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
	    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
	    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
	    'labels_mask': tf.placeholder(tf.int32),
	    'dropout': tf.placeholder_with_default(0., shape=()),
	    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
	}
	
	
	# Create model
	model = model_func(placeholders, input_dim=features[2][1], logging=True)

	# Initialize session
	sess = tf.Session()
	

	
	
	# Define model evaluation function
	def evaluate(features, support, labels, mask, placeholders):
	    t_test = time.time()
	    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
	    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
	    return outs_val[0], outs_val[1], (time.time() - t_test)
	
	# Init variables
	sess.run(tf.global_variables_initializer())
	
	cost_val = []
	
	# Train model

	
	for epoch in range(FLAGS.epochs):
	
	    t = time.time()
	    # Construct feed dictionary
	    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
	    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
	
	    # Training step
	    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
	
	    # Validation
	    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
	    cost_val.append(cost)
	
	    # Print results
	    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
	          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
	          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
	
	    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
	        print("Early stopping...")
	        break
	
	print("Optimization Finished!")

	# Testing
	test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
	print("Test set results:", "cost=", "{:.5f}".format(test_cost),
	      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
	
	feed_dict_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
	y_pred = sess.run([model.predict()],feed_dict_val)[0]
	
	sess.close()
	
	
	
	
	return y_pred









def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_directed_data(path="data/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
    features = sp.csr_matrix(
            idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)


    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    #features = features.toarray()
    adj = np.array(adj.todense(), dtype=np.float64 )

    idx_train = np.array(idx_train, dtype=np.int32)
    idx_val = np.array(idx_val, dtype=np.int32)
    idx_test = np.array(idx_test, dtype=np.int32)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



def load_data(dataset_str, complete=False, random_graph=False, featureless=False, randomized_features=False, knn_n=None):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer' or dataset_str == 'nell.0.1' \
            or dataset_str == 'nell.0.01' or dataset_str == 'nell.0.001' or dataset_str == 'wikipedia':
        # Fix citeseer and NELL datasets (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        # For NELL, we need to add some zero-vecs between the end of allx and the beginning of tx
        zero_mat_x = sp.lil_matrix((min(test_idx_reorder) - allx.shape[0], x.shape[1]))
        zero_mat_y = np.zeros((min(test_idx_reorder) - allx.shape[0], y.shape[1]))
        features = sp.vstack((sp.vstack((allx, zero_mat_x)), tx))
        labels = np.vstack((np.vstack((ally, zero_mat_y)), ty))
        # We also need to add some zero-vecs at the end of tx to match adj
        zero_mat_x = sp.lil_matrix((len(graph)-features.shape[0], x.shape[1]))
        zero_mat_y = np.zeros((len(graph)-features.shape[0], y.shape[1]))
        features = sp.vstack((features, zero_mat_x)).tolil()
        labels = np.vstack((labels, zero_mat_y))
    else:
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))

    features[test_idx_reorder, :] = features[test_idx_range, :]
    if complete:
        graph = turn_graph_to_complete(graph)
    elif random_graph:
        graph = turn_graph_to_random(graph)
    elif featureless:
        features = one_hot_features(features)
    elif randomized_features:
        features = random_features(features)
    elif knn_n is not None:
        graph = create_knn_graph(allx, tx, knn_n)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    if len(ally) > 500:
        idx_val = range(len(y), len(y)+500)
    else:
        idx_val = range(len(y), int(len(y)*1.30))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
	

		
		
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


#functions from networkx 2.3
def directed_laplacian_matrix(G, nodelist=None, weight='weight',
                              walk_type=None, alpha=0.95):
    r"""Returns the directed Laplacian matrix of G.

    The graph directed Laplacian is the matrix

    .. math::

        L = I - (\Phi^{1/2} P \Phi^{-1/2} + \Phi^{-1/2} P^T \Phi^{1/2} ) / 2

    where `I` is the identity matrix, `P` is the transition matrix of the
    graph, and `\Phi` a matrix with the Perron vector of `P` in the diagonal and
    zeros elsewhere.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy array
      Normalized Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    See Also
    --------
    laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    import scipy as sp
    from scipy.sparse import spdiags, linalg

    P = _transition_matrix(G, nodelist=nodelist, weight=weight,
                           walk_type=walk_type, alpha=alpha)

    n, m = P.shape

    evals, evecs = linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = sp.sqrt(p)
    Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
    I = sp.identity(len(G))

    return I - (Q + Q.T) / 2.0



def directed_combinatorial_laplacian_matrix(G, nodelist=None, weight='weight',
                                            walk_type=None, alpha=0.95):
    r"""Return the directed combinatorial Laplacian matrix of G.

    The graph directed combinatorial Laplacian is the matrix

    .. math::

        L = \Phi - (\Phi P + P^T \Phi) / 2

    where `P` is the transition matrix of the graph and and `\Phi` a matrix
    with the Perron vector of `P` in the diagonal and zeros elsewhere.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy array
      Combinatorial Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    See Also
    --------
    laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    from scipy.sparse import spdiags, linalg

    P = _transition_matrix(G, nodelist=nodelist, weight=weight,
                           walk_type=walk_type, alpha=alpha)

    n, m = P.shape

    evals, evecs = linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    Phi = spdiags(p, [0], n, n)

    Phi = Phi.todense()

    return Phi - (Phi*P + P.T*Phi) / 2.0


def _transition_matrix(G, nodelist=None, weight='weight',
                       walk_type=None, alpha=0.95):
    """Returns the transition matrix of G.

    This is a row stochastic giving the transition probabilities while
    performing a random walk on the graph. Depending on the value of walk_type,
    P can be the transition matrix induced by a random walk, a lazy random walk,
    or a random walk with teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    P : NumPy array
      transition matrix of G.

    Raises
    ------
    NetworkXError
        If walk_type not specified or alpha not in valid range
    """

    import scipy as sp
    from scipy.sparse import identity, spdiags
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = "random"
            else:
                walk_type = "lazy"
        else:
            walk_type = "pagerank"

    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    n, m = M.shape
    if walk_type in ["random", "lazy"]:
        DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)
        if walk_type == "random":
            P = DI * M
        else:
            I = identity(n)
            P = (I + DI * M) / 2.0

    elif walk_type == "pagerank":
        if not (0 < alpha < 1):
            raise nx.NetworkXError('alpha must be between 0 and 1')
        # this is using a dense representation
        M = M.todense()
        # add constant to dangling nodes' row
        dangling = sp.where(M.sum(axis=1) == 0)
        for d in dangling[0]:
            M[d] = 1.0 / n
        # normalize
        M = M / M.sum(axis=1)
        P = alpha * M + (1 - alpha) / n
    else:
        raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")

    return P
