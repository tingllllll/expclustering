import numpy as np
import sklearn
import itertools
from collections import Counter

def l2_squared(left, right=None, cuda=True):
    if cuda:
        from _L2_ImplCuda import cuda_l2_squared
        return cuda_l2_squared(left, right)
    else:
        from _L2_ImplNumba import numba_l2_squared
        return numba_l2_squared(left, right)

def knn_affinity(X, sigma, k):
    n_instance = X.shape[0]
    dist_sqs = l2_squared(X,None, False)
    thresholds = np.zeros(n_instance)
    for i in range(n_instance):
        thresholds[i] = sorted(dist_sqs[i])[k]
    kernel_weights = np.exp(-dist_sqs/sigma)
    A_dense = (kernel_weights + np.transpose(kernel_weights)) / 2
    A = A_dense.copy()
    for i in range(n_instance):
        for j in range(i+1, n_instance):
            if dist_sqs[i,j] > thresholds[i] and dist_sqs[i,j] > thresholds[j]:
                A[i,j] = 0
                A[j,i] = 0
    return A, A_dense




def knn_cosine_sim(X, k):
    print(np.shape(X))
    num_points = X.shape[0]
    A_dense = sklearn.metrics.pairwise.cosine_similarity(X, X)
   
    A = A_dense.copy()
    thresholds = np.zeros(num_points)
    for i in range(num_points):
        thresholds[i] = sorted(A[i], reverse=True)[k]

    for i in range(num_points):
        for j in range(num_points):
            if A[i,j] < thresholds[i] and A[j,i] < thresholds[j]:
                A[i,j] = 0
            # dist_ij = pd[i,j]
            # w = np.exp(-1 * dist_ij ** 2 / np.clip((sigma*sigma), a_min=1e-10, a_max=None))
            # if dist_ij < thresholds[i] or dist_ij < thresholds[j]:
            #     A[i,j] = w
            #     A[j,i] = w
    return A, A_dense 

# taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def transitive_closure(ml, cl, n):
    
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()
    
 
    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)
    

    def dfs(i, graph, visited, component):
     
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)
        

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)

    
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' %(i, j))

    return ml_graph, cl_graph
    
    
def transitive(ml,  n):
    
    ml_graph = dict()

    for i in range(n):
        ml_graph[i] = set()
    
 
    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)
    

    
    return ml_graph
    

def generate_constraints_label(y, N_ML, N_CL, A_dense):
    n_instance = y.shape[0]
    mls = []
    cls = []
    #mls表示真实标签相同的聚类  cls表示标签不相同的标签聚类
    while len(mls) < N_ML:
        while True:
            i, j = np.random.randint(0, n_instance, size=2)
            if i != j:
                break
        if y[i] == y[j]:
            mls.append([i,j])
    while len(cls) < N_CL:
        while True:
            i, j = np.random.randint(0, n_instance, size=2)
            if i != j:
                break
        if y[i] != y[j]:
            cls.append([i,j])
    
    # Make np array and ensure valid 2D shapes even if empty
    
    cls = np.array(cls, dtype=int).reshape((-1, 2))
    mls = np.array(mls, dtype=int).reshape((-1, 2))
    
    ml_graph, cl_graph = transitive_closure(mls, cls, n_instance)
    
    mtx_con = np.zeros([n_instance,n_instance])
    ml_counts = 0
    cl_counts = 0
    for i in ml_graph.keys():
        for j in ml_graph[i]:
            mtx_con[i, j] = 1
            mtx_con[j, i] = 1
            ml_counts += 2
    for i in cl_graph.keys():
        for j in cl_graph[i]:
            mtx_con[i, j] = -1
            mtx_con[j, i] = -1
            cl_counts += 2
    # mtx_con[mtx_con == -1] = - ml_counts / cl_counts
    # set weight
    A_dense = A_dense - np.diag(np.diag(A_dense))
    # A_dense_max = np.max(A_dense[np.where(A_dense>0)])
    # A_dense_min = np.min(A_dense[np.where(A_dense>0)])

    A_dense_max = np.max(A_dense)
    A_dense_min = np.min(A_dense)
    # print(A_dense_max, A_dense_min)
    for i in range(mtx_con.shape[0]):
        for j in range(mtx_con.shape[1]):
            if mtx_con[i, j] > 0:
                mtx_con[i, j] = A_dense_max - A_dense[i,j]
            if mtx_con[i, j] < 0:
                mtx_con[i, j] = (A_dense_min - A_dense[i, j])*(ml_counts / cl_counts)
    return mtx_con
    
def generate_constraints_explainable(y, X, N_ML, N_CL, A_dense):
    n_instance = y.shape[0]
    n_d=X.shape[1]
    
    #统计类别个数
    a = Counter(y)
    class_min=np.min(list(dict(a).values()))

    mls = []
    cls = []
    mls_value=[]
    
    #对样本每个维度排序，并且返回坐标
    X_d=[]
    y_d_ind=[] #某个维度y的排序
    ind_d=[]
    
    for j in range(n_d):
       X_d.append(sorted(X[:,j])) #第j个维度数据排序
       ind=np.argsort(X[:,j])  #第j个维度数据排序返回索引
       ind_d.append(ind)
       y_d_ind.append(y[ind])
       
    
    while len(mls) < N_ML:
        j = int(np.random.randint(0, n_d, size=1)) #随机挑选一个维度
        connectpoints_list=[len(list(v)) for k,v in itertools.groupby(y_d_ind[j])] #对这个维度连续相同元素排序的结果
 
        if np.max(connectpoints_list)>=class_min*1.8:
            max_cps_ind=np.argmax(connectpoints_list) #第j维标签排序中 最多的元素从connectpoints_list列表的哪个位置开始
            ind_cp_start=int(np.sum(connectpoints_list[0:max_cps_ind]))#最多的元素从那个什么位置开始
            y_j=ind_d[j] #取j维索引
            ind_mls=y_j[ind_cp_start:ind_cp_start+np.max(connectpoints_list)]
            
            bb = list(itertools.permutations(ind_mls, 2))
            for b in range(len(bb)):
               i,j=bb[b]
               mls.append([i,j])
            
    
    # Make np array and ensure valid 2D shapes even if empty
    
    #cls = np.array(cls, dtype=int).reshape((-1, 2))
    mls = np.array(mls, dtype=int).reshape((-1, 2))
    
    ml_graph, cl_graph = transitive_closure(mls, cls, n_instance)
    #ml_graph, cl_graph = transitive(mls, n_instance)
    
    mtx_con = np.zeros([n_instance,n_instance])
    ml_counts = 0
    for i in ml_graph.keys():
        for j in ml_graph[i]:
            mtx_con[i, j] = 1
            mtx_con[j, i] = 1
            ml_counts += 2
    
                
    return mtx_con





if __name__=='__main__':
    y = np.array([0,1,2,1,0,1,2,1,0])
    N = 5
    A_dense = np.zeros([y.shape[0],y.shape[0]])
    A_dense[0,0] = 1
    A_dense[-1,-1] = -1
    mtx_con = generate_constraints_pairwise(y,N,N,A_dense)
    print(np.sum(mtx_con!=0))
    # A_CL = generate_constraints_bool(y, N, "CL")
    # A_ML = generate_constraints_bool(y, N, "ML")
    # print(A_ML)
    # print(A_CL)
    # A_ML = transitive_enclosure_ML(A_ML)
    # A_CL = entailment(A_ML, A_CL)
    # print(A_ML)
    # print(A_CL)


