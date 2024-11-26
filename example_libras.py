import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.cluster import KMeans

from queue import Queue

import scipy
from graph_construction import knn_affinity, generate_constraints_label, generate_constraints_explainable
from SSE_expclustering import FlatSSE
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,silhouette_score,accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from utils import PartitionTree
import argparse
import csv
import pandas as pd


#parser = argparse.ArgumentParser()
#parser.add_argument('--method', required=True, choices=['SSE_partitioning_pairwise', 'SSE_partitioning_label','SSE_partitioning_bio_pairwise', 'SSE_partitioning_bio_label', 'SSE_hierarchical'])
#parser.add_argument('--dataset', required=True)
#parser.add_argument('--constraint_ratio', type=float, required=True)
#parser.add_argument('--label_weight', default=3.0, type=float)
#parser.add_argument('--explainable_weight', default=2.0, type=float)
#parser.add_argument('--sigmasq', default=100, type=float, help='square of Gaussian kernel band width, i.e., sigma^2')
#parser.add_argument('--exp_repeats', default=100, type=int)
#parser.add_argument('--knn_constant', default=5, type=float)

# for hierarchical clustering
#parser.add_argument('--hie_knn_k', default=5)
#args = parser.parse_args()



def SSE_explainable_clustering(X,n_cluster,j):
    #data_f = scipy.io.loadmat(path)
    #X = np.array(data_f['fea']).astype(float)
    #X = MinMaxScaler().fit_transform(X)

    #get reference label
    kmeans = KMeans(n_clusters=n_cluster, n_init="auto").fit(X)
    y=kmeans.labels_
 
    #y_true = np.array(data_f['gnd']).astype(float).squeeze()
    #n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    sigmasq=100
    knn_constant=n_cluster
    constraint_ratio=0.2
    label_weight=0
    explainable_weight=j
    #knn_k = knn_k_estimating(n_cluster, n_instance, args.knn_constant)
    A, A_dense = knn_affinity(X, sigmasq, knn_constant)
    
    ARIs = []
    NMIs = []
    #for _ in range(args.exp_repeats):
        
        
    A_label = generate_constraints_label(y, int(A.shape[0] * constraint_ratio), int(A.shape[0] * constraint_ratio), A_dense)
    A_explainable = generate_constraints_explainable(y, X, int(A.shape[0] * constraint_ratio), int(A.shape[0] * constraint_ratio), A_dense)
    flatSSE = FlatSSE(A, label_weight*A_label, explainable_weight*A_explainable, len(np.unique(y)))
    y_pred_flatSSE = flatSSE.build_tree()
    n_cluster = np.unique(y_pred_flatSSE)

 
   
    #ARI = adjusted_rand_score(y_true, y_pred_flatSSE)
    #NMI = normalized_mutual_info_score(y_true, y_pred_flatSSE)
    #ARIs.append(ARI)
    #NMIs.append(NMI)
    #print(path, ARI, NMI)
    #print("average: {}\t{}\t{}\n".format(args.dataset, np.mean(ARIs), np.mean(NMIs)))
    
    return y_pred_flatSSE



if __name__=='__main__':
    

  
    for data in ['movement_libras_5.data']: 
        
        #get .mat datasets
        path = f"./data/{data}"
     
        data_f = pd.read_csv(path)
        data_f=np.array(data_f).astype(float)
        d=data_f.shape[1]
        X=data_f[:,0:d]
        X = MinMaxScaler().fit_transform(X)
        y_true=data_f[:,-1]
        n_instance = y_true.shape[0]
        n_cluster = np.unique(y_true).shape[0]

        
               
        
        ARIs_ent = []
        NMIs_ent = []
      
        
  
        for _ in range(10):

                #get SSE_expclustering predict label
                y_pred_ent=SSE_explainable_clustering(X,n_cluster,j)
                ARIs_ent.append(adjusted_rand_score(y_true, y_pred_ent))
                NMIs_ent.append(normalized_mutual_info_score(y_true, y_pred_ent))
     
                print("sse")

             
                with open('result_gamma.csv', 'a+') as file:
                # 创建csv.writer对象
                    writer = csv.writer(file)
                    
                    # 写入表头
                    writer.writerow(['data','method','k', 'ARI', 'NMI']) 

                    # 逐行写入数据
                    writer.writerow([data,'entropy',n_cluster, np.mean(ARIs_ent), np.mean(NMIs_ent)])
                  


    
    
  