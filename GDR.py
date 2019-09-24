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
import scipy as sc
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from utils import *


class GDR(object):
    """
    Class to compute the class for nodes given a semi-supervised graph dataset

    Inputs:
        dataset:
            String that describes the dataset e.g. 'cora','citeseer','pubmed',
            'wikipedia'

        tpe_rw:
            String that describes the type of Laplacian
                'Combinatorial' or 'Normalised'
                
    """
    def __init__(self, dataset, tpe_rw):
        
        self.dataset = dataset # string of dataset     
        self.tpe_rw = tpe_rw #type of random walk: Combinatoria, Normalised, Discrete
        
                    
        # Data to be loaded with load_ dataset
        self.A = [] #adjacency matrix
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.feature_data = []
        self.feature_data_orig = []
        
        self.directed = False
        
        # Prior distribution to be calculated with prior_distribution
        self.similarity = []  #collect the similarity vectors after preconditioning
        self.similarity_orig = [] #collec the original similarity vectors
        

     
    def load_dataset(self):
        """
        Extracting data from datasets and loading to the class
        """
        if self.dataset == 'cora-dr' or self.dataset == 'cora-d':
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_directed_data()
            self.directed = True
        else:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(self.dataset)

        if self.dataset == 'cora-dr': #reverse the direction
            self.A = sc.sparse.csr_matrix(adj, dtype=np.float64).T     
        else:
            self.A = sc.sparse.csr_matrix(adj, dtype=np.float64)
            


        self.feature_data      = sc.sparse.csr_matrix(features, dtype=np.float64)
        self.feature_data_orig = self.feature_data.copy()

        self.train_data = sc.sparse.csr_matrix(y_train, dtype=np.float64)
        self.test_data  = sc.sparse.csr_matrix(y_test, dtype=np.float64)
        self.val_data   = sc.sparse.csr_matrix(y_val, dtype=np.float64)

       

    def define_comm_feature(self):
        """
        Embedding the community labels in feature space
        """

        self.comm_features = self.train_data.T.dot(self.feature_data)
        norm = self.comm_features.sum(axis=1)
        norm[norm==0] = 1
        self.comm_features = sc.sparse.csr_matrix(self.comm_features/norm) #normalise for each node
            
 
    def define_similarity_distribution(self, tpe = 'cos'):
        """
        compute the prior probability distribution from the training data and the feautres
        """

        if tpe == 'mlp': #use MLP for the prior distribution
            if self.dataset == 'cora-d' or self.dataset == 'cora-dr':
                self.similarity_orig = define_MLP_distribution('cora', directed = True)
            else:
                self.similarity_orig = define_MLP_distribution(self.dataset)
            
        elif tpe == 'cos':
            #calculating the cosine similarity between each node and the features that describe each community
            self.similarity_orig = cosine_similarity(self.feature_data.toarray(), self.comm_features.toarray())

        # Scaling the similarity of each node to 1, i.e. probability distribution
        norm = np.tile(self.similarity_orig.sum(axis=1), (self.comm_features.shape[0],1)).T
        norm[norm==0] = 1 #if not probabilities, don't divide
        self.similarity_orig  = sc.sparse.csr_matrix( self.similarity_orig / norm )
                       

    def precondition_laplacian(self, tpe = 'weighted' , disp = True):
        """
        Compute a graph laplacian on the class space to smooth the prior distrubution
        """

        G = nx.complete_graph(np.shape(self.similarity_orig)[1])

        if tpe == 'weighted': #weight the preconditioning by similarities
            for i, j in G.edges():
                G[i][j]['weight'] = float(cosine_similarity([self.comm_features.toarray()[i,:],],[self.comm_features.toarray()[j,:],]))
        else: 
            for i, j in G.edges():
                G[i][j]['weight'] = 1.
                
        self.L_precond = nx.laplacian_matrix(G) #combinatorial Laplacian matrix
    
        #normalise with the second smallest eigavalue, so we always get relaxation time of 1 
        if np.shape(self.L_precond.toarray())[0]>3:
            self.L_precond /= abs(sc.sparse.linalg.eigs(self.L_precond, which='SM', k=2)[0][1])
        else:
            print('implement spectral gap for small 3 feature (Pubmed)')
 

    def precondition_distribution(self, d_t , disp=True):
        """
        Predonditioning the prior distribution by smoothing it with a graph Laplacian
        """
       
        exp = sc.linalg.expm(-d_t*self.L_precond.toarray())
        node_label_dist = self.similarity_orig.toarray()#.dot(exp)
        H = self.train_data.toarray().copy()
        
        #Final Distribution sets the training nodes to their known probability distribution
        # setting non-training nodes to their feature probability
        indices = np.where(~H.any(axis=1))[0]
        for row in indices:
            H[row] = node_label_dist[row]
        #if any rows have nan then we set them to equal probability across labels
        index_nan = np.where(np.isnan(H[:,0]))[0]
        for row in index_nan:
            H[row] = 1/np.shape(H)[1]
            
        self.similarity = H.copy()

        # Now automatically calculate how accurate this prior distribution is
        class_H = np.zeros_like(H)
        class_H[np.arange(len(H)), H.argmax(1)] = 1
        (test_y_pred, test_y_real, test_accuracy) = accuracy_predictions(class_H, self.test_data)
        (val_y_pred, val_y_real, val_accuracy) = accuracy_predictions(class_H, self.val_data)
        
        if disp:
            print('----------------------------------')
            print('Accuracy of prior cosine similarity distribution:')
            print('Val accuracy score: ' + str(val_accuracy))
            print('Test accuracy score: ' + str(test_accuracy))
            print('----------------------------------')
    
        return val_accuracy, test_accuracy
    
    def Laplacian(self):
        """
        Create the Laplacian matrix for the continuous random walk
        """
    

        if self.directed: #uses the networkx 2.3 functions copied in utils.py
            if self.tpe_rw == 'Combinatorial':
                L  = directed_combinatorial_laplacian_matrix(nx.DiGraph(self.A), walk_type='pagerank', alpha=0.85)
            elif self.tpe_rw == 'Normalized':
                L  = directed_laplacian_matrix(nx.DiGraph(self.A), walk_type='pagerank', alpha=0.1)
               
        else:
            G = nx.Graph(self.A)

            graphs = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)

            if self.tpe_rw == 'Combinatorial':
                #combinatorial random walk
                L = nx.laplacian_matrix(G)
                self.v = np.zeros(len(G))
                for graph in graphs:
                    self.v[list(graph.nodes)] = np.ones(len(graph.nodes))/len(graph.nodes)
             
        
            elif self.tpe_rw == 'Normalized':
                #normalized random walk
                degree = np.array(self.A.sum(1)).flatten()
                L = sc.sparse.csr_matrix(nx.laplacian_matrix(G).toarray().dot(np.diag(1./degree)))

                self.v = np.zeros(len(G))
                for graph in graphs:
                    self.v[list(graph.nodes)] = degree[list(graph.nodes)]/degree[list(graph.nodes)].sum()*0.5
             

            elif self.tpe_rw == 'max_entropy':
                #maximum entropy random walk
                self.v = np.zeros(len(G))
                L = np.zeros([len(G),len(G)])
                A = nx.adjacency_matrix(G).toarray()

                for graph in graphs:
                    A_tmp = A[np.ix_(list(graph.nodes), list(graph.nodes))]
                    eigs = sc.sparse.linalg.eigsh(1.*A_tmp, which='LM', k=1)
                    psi = eigs[1][:,0]
                    self.v[list(graph.nodes)] = psi**2
                    lamb_0 = abs(eigs[0][0])
                    L[np.ix_(list(graph.nodes),list(graph.nodes))] = np.eye(len(graph.nodes)) - np.diag(psi).dot(A_tmp).dot(np.diag(1./psi))/lamb_0
             
                L = sc.sparse.csc_matrix(L)


        #normalise with the second smallest eigenvalue, so we always get relaxation time of 1 (for the largest connected component)
        if self.directed:
            L_sub = L

        else: #get the largest connected component first
            graphs = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
            print(len(graphs), 'connected components, we rescale L with the largest one only')

            L_sub = sc.sparse.csc_matrix(L.toarray()[np.ix_(graphs[0].nodes, graphs[0].nodes)])

        l_2 = abs(sc.sparse.linalg.eigs(L_sub, which='SM', k=2)[0][1])

        print("l_2 = ", l_2)
    
        self.L  = L / l_2

        
    def apply_exponential(self, disp=True):
        """
        Compute the propagation of labels across the network
        """
        
        self.H_list = np.array(sc.sparse.linalg.expm_multiply( -self.L, self.similarity, self.t_min, self.t_max, self.N_t))
        self.times = np.linspace(self.t_min, self.t_max, self.N_t)

    
    def label_prediction_prior(self, disp = False):
        best_classes_prior = np.argmax(self.similarity, axis=1)# np.unravel_index(np.argmax(node_class_tmp, axis=1), node_class_tmp.shape)
        
        class_H_prior = np.zeros_like(self.similarity) #array to collect the predicted classes
        class_H_prior[np.arange(len(best_classes_prior)), best_classes_prior] = 1 #finally, set the predicted classes

        #compute the prior accuracies
        (train_y_pred,train_y_real,train_accuracy) = accuracy_predictions(class_H_prior, self.train_data.toarray())
        (val_y_pred,val_y_real,val_accuracy) = accuracy_predictions(class_H_prior, self.val_data.toarray())
        (test_y_pred,test_y_real,test_accuracy) = accuracy_predictions(class_H_prior, self.test_data.toarray())

        if disp:
            print('Prior distribution before random-walk:')
            print('Training accuracy score: ' + str(train_accuracy))
            print('Validation accuracy score: ' + str(val_accuracy))
            print('Testing accuracy score: ' + str(test_accuracy))

        return val_accuracy, test_accuracy

    def label_prediction_overshoot(self, H_list, disp=True):
         
        """
        find the best class of each node from overshooting
        """
 
        #stationary state of each class 
        G = nx.Graph(self.A)
        graphs = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
        H_list_inf = np.zeros([len(G), np.shape(self.similarity)[1]])
        for graph in graphs:
            m = self.similarity[list(graph.nodes)].sum(0)
            H_list_inf[list(graph.nodes)] = self.v[list(graph.nodes), np.newaxis]*m[np.newaxis,:]

        overshoots = np.max(H_list - H_list_inf[np.newaxis, :, :], axis=0) #find the overshooting value of each class/node
            
        H_list_no_overshoots = H_list.copy()
        H_list_no_overshoots[:, overshoots <= 0 ] = 0 #set to 0 the non-overshooting classes
        
        no_os_id = np.argwhere(H_list_no_overshoots[0,:,:].all(axis=1) == 0) #find the nodes with no overshooting classes
        class_0 = H_list[0].argmax(1) #class prediction at time 0 (min time)
        H_list_no_overshoots[:, no_os_id, class_0[no_os_id] ] = 1 #set the best class of to be the initial one

        node_class_tmp = np.max(H_list_no_overshoots, axis=0) #find the max values accross all times
        
        best_classes = np.argmax(node_class_tmp, axis=1)# np.unravel_index(np.argmax(node_class_tmp, axis=1), node_class_tmp.shape)
        
        class_H = np.zeros_like(H_list[0]) #array to collect the predicted classes
        class_H[np.arange(len(best_classes)), best_classes] = 1 #finally, set the predicted classes
         
        #set the training nodes back
        class_H[self.train_data.toarray().any(axis=1)>0] = self.train_data.toarray()[self.train_data.toarray().any(axis=1)>0]
        
        #compute the accuracies
        (train_y_pred,train_y_real,train_accuracy) = accuracy_predictions(class_H, self.train_data.toarray())
        (val_y_pred,val_y_real,val_accuracy) = accuracy_predictions(class_H, self.val_data.toarray())
        (test_y_pred,test_y_real,test_accuracy) = accuracy_predictions(class_H, self.test_data.toarray())

        if disp:
            print('Cosine Similarity prior distribution after random-walk:')
            print('Training accuracy score: ' + str(train_accuracy))
            print('Validation accuracy score: ' + str(val_accuracy))
            print('Testing accuracy score: ' + str(test_accuracy))
 
        return val_accuracy, test_accuracy


    def scan_tmin(self, n_min, output_full = True):
        """
        scan for t_min, and find the best one from validation set
        """

        Val = np.zeros(n_min)
        Test = np.zeros(n_min)

        val_prior, test_prior = self.label_prediction_prior() 

        for i in range(n_min):
            val, test = self.label_prediction_overshoot(self.H_list[i:], disp=False)

            Val[i] = val
            Test[i] = test
            
        if output_full:
            return  np.argmax(Val), Val, Test, val_prior, test_prior
        else:
            return Val[np.argmax(Val)], Test[np.argmax(Val)], val_prior, test_prior
