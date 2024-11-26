import numba as nb
import heapq
import numpy as np

from SE_partitioning import Graph, Edge, read_graph, get_graph

EPS = 1e-15

@nb.jit(nopython=True)
def cal_module_SSE(g, sum_degrees, v, v_minus):
    return -(g/sum_degrees) * np.log2(v/v_minus)

class FlatSSE():
    def __init__(self, A, A_label,A_explainable, num_cluster=None, mustlink_first_L = None,mustlink_first_E = None):
        self.mustlink_first_L = mustlink_first_L
        self.mustlink_first_E = mustlink_first_E
        self.num_cluster = num_cluster
        A = A - np.diag(np.diag(A))
        A_label = A_label - np.diag(np.diag(A_label)) 
        A_explainable = A_explainable - np.diag(np.diag(A_explainable))
        self.A = A
        self.A_label = A_label
        self.A_explainable = A_explainable
        self.graph = get_graph(A)
        self.graph_label = get_graph(A_label)
        self.graph_explainable = get_graph(A_explainable)
        self.SSE = 0
        self.communities = dict()  #每个节点的信息 记录方式(nodes, volume, cut, cut', SSE)
        self.pair_cuts = dict()   #键：边  值：边的权重  
        self.pair_cuts_label = dict()
        self.pair_cuts_explainable = dict()
        self.connections = dict()  #邻接节点
        self.connections_label = dict()
        self.connections_explainable = dict()
        for i in range(self.graph.num_nodes):
            self.connections[i] = set()
            self.connections_label[i] = set()
            self.connections_explainable[i] = set()

    def init_encoding_tree(self):
        for i in range(self.graph.num_nodes):
            if self.graph.node_degrees[i] == 0:
                continue
            SSEi = - ((self.graph.node_degrees[i]) / self.graph.sum_degrees) * np.log2(
                self.graph.node_degrees[i] / self.graph.sum_degrees)   #计算每个节点的度数
            # SSEi = - ((self.graph.node_degrees[i] + self.graph_con.node_degrees[i])/self.graph.sum_degrees) * np.log2(self.graph.node_degrees[i]/self.graph.sum_degrees)
            ci = ({i}, self.graph.node_degrees[i], self.graph.node_degrees[i], self.graph_label.node_degrees[i],self.graph_explainable.node_degrees[i],SSEi)  # nodes, volume, cut, cut', SSE
            self.communities[i] = ci 
            self.SSE += SSEi
        for i in self.graph.adj.keys():  #self.graph.adj表示节点的邻接边   key表示点
            for edge in self.graph.adj[i]:
                self.pair_cuts[frozenset([edge.i, edge.j])] = edge.weight
                self.connections[i].add(edge.j)
                self.connections[edge.j].add(i)
        for i in self.graph_label.adj.keys():
            for edge in self.graph_label.adj[i]:
                self.pair_cuts_label[frozenset([edge.i, edge.j])] = edge.weight
                self.connections_label[edge.i].add(edge.j)
                self.connections_label[edge.j].add(edge.i)
        for i in self.graph_explainable.adj.keys():
            for edge in self.graph_explainable.adj[i]:
                self.pair_cuts_explainable[frozenset([edge.i, edge.j])] = edge.weight
                self.connections_explainable[edge.i].add(edge.j)
                self.connections_explainable[edge.j].add(edge.i)

    def merge_deltaH_SSE_old(self, commID1, commID2):
        vertices1, v1, g1, g_label1, g_explainable1, SSE1 = self.communities.get(commID1)
        vertices2, v2, g2, g_label2, g_explainable2, SSE2 = self.communities.get(commID2)
        vx = v1 + v2
        gx = g1 + g2 - 2*self.pair_cuts.get(frozenset([commID1, commID2]))
        g_labelx = g_label1 + g_label2
        g_explainablex=g_explainable1+g_explainable2
       
        if frozenset([commID1,commID2]) in self.pair_cuts_label:
            g_labelx -= 2*self.pair_cuts_label.get(frozenset([commID1, commID2]))
        if frozenset([commID1,commID2]) in self.pair_cuts_explainable:
            g_explainablex -= 2*self.pair_cuts_explainable.get(frozenset([commID1, commID2]))    
            
        SSEx = cal_module_SSE(gx+g_labelx+g_explainablex, self.graph.sum_degrees, vx, self.graph.sum_degrees)
        for vertex in vertices1:
            SSEx += cal_module_SSE(self.graph.node_degrees[vertex], self.graph.sum_degrees,
                                   self.graph.node_degrees[vertex], vx)
        for vertex in vertices2:
            SSEx += cal_module_SSE(self.graph.node_degrees[vertex], self.graph.sum_degrees,
                                   self.graph.node_degrees[vertex], vx)
        return SSE1+SSE2-SSEx

    def merge_deltaH_SSE(self, commID1, commID2):
        vertices1, v1, g1, g_label1, g_explainable1, SSE1 = self.communities.get(commID1)
        vertices2, v2, g2, g_label2, g_explainable2, SSE2 = self.communities.get(commID2)
        vx = v1 + v2
        gx = g1 + g2 - 2 * self.pair_cuts.get(frozenset([commID1, commID2]))
        
        g_labelx = g_label1 + g_label2
        g_explainablex = g_explainable1 + g_explainable2
        
        if frozenset([commID1, commID2]) in self.pair_cuts_label:
            g_labelx -= 2 * self.pair_cuts_label.get(frozenset([commID1, commID2]))
        if frozenset([commID1, commID2]) in self.pair_cuts_explainable:
            g_explainablex -= 2 * self.pair_cuts_explainable.get(frozenset([commID1, commID2]))
            
        deltaH = (v1-g1-g_label1-g_explainable1)*np.log2(v1) + (v2-g2-g_label2-g_explainable2)*np.log2(v2) - (vx-gx-g_labelx-g_explainablex)*np.log2(vx)
        deltaH += (g1+g_label1+g_explainable1+g2+g_label2+g_explainable2-gx-g_labelx-g_explainablex)*np.log2(self.graph.sum_degrees)
        deltaH /= self.graph.sum_degrees
        return deltaH

    def refinement_SSE(self):
        y, _ = self.communities2label()
        adj = self.A
        adj -= np.diag(np.diag(adj))
        W_label = self.A_label
        W_label -= np.diag(np.diag(W_label)).copy()
        W_explainable = self.A_explainable
        W_explainable-= np.diag(np.diag(W_explainable)).copy()
        tol = 1e-20
        max_iter = 300
        if y is None:
            n, k = adj.shape[0], 3
            y = np.random.randint(k, size=n)
        else:
            n, k = adj.shape[0], np.amax(y) + 1

        W = np.array(adj.copy(), dtype=np.float64)
        D = np.diag(np.sum(W, axis=-1, keepdims=False))
        D_label = np.diag(np.sum(W_label, axis=-1, keepdims=False))
        D_explainable = np.diag(np.sum(W_explainable, axis=-1, keepdims=False))
        S = np.eye(k)[y.reshape(-1)].astype(np.float64)     # one hot of y
        volW = np.sum(W, dtype=np.float64)      # sum of degrees of A
        links_mtx = np.matmul(np.matmul(S.T, W), S)
        degree_mtx = np.matmul(np.matmul(S.T, D), S)
        links_label_mtx = np.matmul(np.matmul(S.T, W_label), S)
        degree_label_mtx = np.matmul(np.matmul(S.T, D_label), S)
        links_explainable_mtx = np.matmul(np.matmul(S.T, W_explainable), S)
        degree_explainable_mtx = np.matmul(np.matmul(S.T, D_explainable), S)
        
        links = np.diagonal(links_mtx).copy()     # asso of each community, i.e., volume - cut
        degree = np.diagonal(np.clip(degree_mtx, a_min=EPS, a_max=None)).copy()
        links_label = np.diagonal(links_label_mtx).copy()
        degree_label = np.diagonal(np.clip(degree_label_mtx, a_min=EPS, a_max=None)).copy()
        links_explainable = np.diagonal(links_explainable_mtx).copy()
        degree_explainable = np.diagonal(np.clip(degree_explainable_mtx, a_min=EPS, a_max=None)).copy()
        # cuts_con = degree_con - links_con
        sses = ((-links+degree_label+degree_explainable-links_label-links_explainable) / volW) * np.log2(np.clip(degree, a_min=1e-10, a_max=None) / volW)
        z = y.copy()
        sse = np.sum(sses)
        for iter_num in range(max_iter):
            for i in range(n):
                zi = z[i]
                links[zi] -= np.matmul(W[i, :], S[:, zi]) + np.matmul(S[:, zi].T, W[:, i])
                degree[zi] -= D[i, i]
                links_label[zi] -= np.matmul(W_label[i,:], S[:,zi]) + np.matmul(S[:,zi].T, W_label[:,i])
                degree_label[zi] -= D_label[i,i]
                links_explainable[zi] -= np.matmul(W_explainable[i,:], S[:,zi]) + np.matmul(S[:,zi].T, W_explainable[:,i])
                degree_explainable[zi] -= D_explainable[i,i]
                
                sses[zi] = ((-links[zi]+degree_label[zi]+degree_explainable[zi]-links_label[zi]-links_explainable[zi]) / volW) * np.log2(np.clip(degree[zi], a_min=1e-10, a_max=None) / volW)
                S[i, zi] = 0
                z[i] = -1

                links_new = links.copy()
                degree_new = degree.copy()
                links_new += np.matmul(W[i, :], S) + np.matmul(W[:, i].T, S)
                degree_new += D[i, i]
                
                links_label_new = links_label.copy()
                degree_label_new = degree_label.copy()
                links_label_new += np.matmul(W_label[i, :], S) + np.matmul(W_label[:, i].T, S)
                degree_label_new += D_label[i,i]
                
                links_explainable_new = links_explainable.copy()
                degree_explainable_new = degree_explainable.copy()
                links_explainable_new += np.matmul(W_explainable[i, :], S) + np.matmul(W_explainable[:, i].T, S)
                degree_explainable_new += D_explainable[i,i]
                
                
                sses_new = ((-links_new+degree_label_new+degree_explainable_new-links_label_new-links_explainable_new) / volW) * np.log2(np.clip(degree_new, a_min=1e-10, a_max=None) / volW)
                delta_sses = sses_new - sses

                opt_i = np.argmax(delta_sses)

                zi = opt_i
                z[i] = zi
                S[i, zi] = 1
                links[zi] = float(links_new[zi])
                degree[zi] = float(degree_new[zi])
                links_label[zi] = float(links_label_new[zi])
                degree_label[zi] = float(degree_label_new[zi])
                links_explainable[zi] = float(links_explainable_new[zi])
                degree_explainable[zi] = float(degree_explainable_new[zi])
                sses[zi] = float(sses_new[zi])
            if np.sum(sses) - sse < tol:
                break
            sse = np.sum(sses)
        # z = self.remove_empty_cluster(z)
        # assert np.max(z)+1 == np.unique(z).shape[0]
        # k = np.max(z) + 1
        return z

    def remove_empty_cluster(self, z):
        z_new = np.zeros_like(z)
        label_old2new = dict()
        for i, label in enumerate(np.unique(z)):
            label_old2new[label] = i
        for i in range(z.shape[0]):
            z_new[i] = label_old2new[z[i]]
        return z_new

    def merge(self):
        merge_queue = []  
        merge_map = dict() 
        
        #搜集合并对信息，并放到队列merge_queue中         
        for pair in self.pair_cuts.keys():
            commID1, commID2 = pair
            if commID1 not in self.communities.keys() or commID2 not in self.communities.keys():
                continue
            deltaH = self.merge_deltaH_SSE(commID1, commID2)  #合并commID1和commID2值的变化
            pair_mustlink_L = 0
            if self.mustlink_first_L:
                if frozenset([commID1,commID2]) in self.pair_cuts_label: #如果当前节点对在关系图中
                    if self.pair_cuts_label[frozenset([commID1,commID2])] > 0:  #在关系图中两个点的熵大于大于0  pair_mustlink = 1
                        pair_mustlink_L = 1
                    elif self.pair_cuts_label[frozenset([commID1,commID2])] < 0:
                        pair_mustlink_L = -1
                    else:
                        pair_mustlink_L = 0
                else:
                    ppair_mustlink_L = 0
            
            pair_mustlink_E = 0            
            if self.mustlink_first_E:
                if frozenset([commID1,commID2]) in self.pair_cuts_explainable: 
                    if self.pair_cuts_explainable[frozenset([commID1,commID2])] > 0:  
                        pair_mustlink_E = 1
                    elif self.pair_cuts_explainable[frozenset([commID1,commID2])] < 0:
                        pair_mustlink_E = -1
                    else:
                        pair_mustlink_E = 0
                else:
                    pair_mustlink_E = 0
                    
            merge_entry = [-pair_mustlink_L,-pair_mustlink_E, -deltaH, pair]
            heapq.heappush(merge_queue, merge_entry)  #将当前节点对的合并信息放入队列
            merge_map[pair] = merge_entry  
        
        
        while len(merge_queue) > 0:
        
            pair_mustlink_L, pair_mustlink_E, deltaH, pair = heapq.heappop(merge_queue)
            pair_mustlink_L = - pair_mustlink_L
            Pair_mustlink_E = - pair_mustlink_E
            deltaH = -deltaH
 
            if pair == frozenset([]):
                continue
            if deltaH<0:
                continue
                
            commID1, commID2 = pair
            if (commID1 not in self.communities) or (commID2 not in self.communities):
                continue
            self.SSE -= deltaH
            comm1 = self.communities.get(commID1)
            comm2 = self.communities.get(commID2)
            
            g_labelx = comm1[3]+comm2[3] 
            g_explainablex = comm1[4]+comm2[4]
            
            if frozenset([commID1,commID2]) in self.pair_cuts_label: 
                g_labelx -= 2*self.pair_cuts_label[frozenset([commID1,commID2])] 
            
            if frozenset([commID1,commID2]) in self.pair_cuts_explainable: 
                g_explainablex -= 2*self.pair_cuts_explainable[frozenset([commID1,commID2])] 
            
            #合并节点1和节点2为新的节点
            new_comm = (comm1[0].union(comm2[0]), comm1[1]+comm2[1], comm1[2]+comm2[2]-2*self.pair_cuts[frozenset([commID1,commID2])],
                        g_labelx,g_explainablex, comm1[5]+comm2[5]-deltaH)
            self.communities[commID1] = new_comm
            self.communities.pop(commID2)

            #----------------------------------------------------------------------------------------
            #如果节点2在节点1的邻接矩阵中，则在1和2的节点中互相删除
            if commID2 in self.connections_label[commID1]:
                self.connections_label[commID1].remove(commID2)
                self.connections_label[commID2].remove(commID1)
                
            #遍历节点1的邻接矩阵，如果在节点2的邻接矩阵中，
            for k in self.connections_label[commID1]:
                if k in self.connections_label[commID2]:
                    self.pair_cuts_label[frozenset([commID1,k])] = self.pair_cuts_label.get(frozenset([commID1,k])) + self.pair_cuts_label.get(frozenset([commID2,k]))
                    self.connections_label[commID2].remove(k)
                    self.connections_label[k].remove(commID2)
                    self.pair_cuts_label.pop(frozenset([commID2,k]))
            
            #遍历关系图节点2的邻接点， 将节点2->邻接节点 换成 节点1->邻接点
            for k in self.connections_label[commID2]:
                self.pair_cuts_label[frozenset([commID1,k])] = self.pair_cuts_label[frozenset([commID2,k])]
                self.pair_cuts_label.pop(frozenset([commID2,k]))
                self.connections_label.get(k).remove(commID2)
                self.connections_label.get(k).add(commID1)
                self.connections_label.get(commID1).add(k)
            self.connections_label.get(commID2).clear()
            
            
            #----------------------------------------------------------------------------------------
            #如果节点2在节点1的邻接矩阵中，则在1和2的节点中互相删除
            if commID2 in self.connections_explainable[commID1]:
                self.connections_explainable[commID1].remove(commID2)
                self.connections_explainable[commID2].remove(commID1)
                
            #遍历节点1的邻接矩阵，如果在节点2的邻接矩阵中，
            for k in self.connections_explainable[commID1]:
                if k in self.connections_explainable[commID2]:
                    self.pair_cuts_explainable[frozenset([commID1,k])] = self.pair_cuts_explainable.get(frozenset([commID1,k])) + self.pair_cuts_explainable.get(frozenset([commID2,k]))
                    self.connections_explainable[commID2].remove(k)
                    self.connections_explainable[k].remove(commID2)
                    self.pair_cuts_explainable.pop(frozenset([commID2,k]))
            
            #遍历关系图节点2的邻接点， 将节点2->邻接节点 换成 节点1->邻接点
            for k in self.connections_explainable[commID2]:
                self.pair_cuts_explainable[frozenset([commID1,k])] = self.pair_cuts_explainable[frozenset([commID2,k])]
                self.pair_cuts_explainable.pop(frozenset([commID2,k]))
                self.connections_explainable.get(k).remove(commID2)
                self.connections_explainable.get(k).add(commID1)
                self.connections_explainable.get(commID1).add(k)
            self.connections_explainable.get(commID2).clear()

            #~~~~~~~~~操心原图的事情~~~~~~~~~~~
            self.connections[commID1].remove(commID2)
            self.connections[commID2].remove(commID1)
            for k in self.connections[commID1]:
                if k in self.connections[commID2]:
                    pair_cut_1k = self.pair_cuts.get(frozenset([commID1,k])) + self.pair_cuts.get(frozenset([commID2,k]))
                    self.pair_cuts[frozenset([commID1,k])] = pair_cut_1k
                    self.connections[commID2].remove(k)
                    self.connections[k].remove(commID2)
                    self.pair_cuts.pop(frozenset([commID2, k]))
                    merge_entry = merge_map.pop(frozenset([commID2, k]))
                    merge_entry[-1] = frozenset([])
                else:
                    pair_cut_1k = self.pair_cuts[frozenset([commID1,k])]
                deltaH1k = self.merge_deltaH_SSE(commID1,k)
                merge_entry = merge_map.pop(frozenset([commID1,k]))
                merge_entry[-1] = frozenset([])
                
                pair_mustlink_L = 0
                if self.mustlink_first_L:
                    if frozenset([commID1,commID2]) in self.pair_cuts_label: #如果当前节点对在关系图中
                        if self.pair_cuts_label[frozenset([commID1,commID2])] > 0:  #在关系图中两个点的熵大于大于0  pair_mustlink = 1
                            pair_mustlink_L = 1
                        elif self.pair_cuts_label[frozenset([commID1,commID2])] < 0:
                            pair_mustlink_L = -1
                        else:
                            pair_mustlink_L = 0
                    else:
                        pair_mustlink_L = 0
            
                pair_mustlink_E = 0            
                if self.mustlink_first_E:
                    if frozenset([commID1,commID2]) in self.pair_cuts_explainable: 
                        if self.pair_cuts_explainable[frozenset([commID1,commID2])] > 0:  
                            pair_mustlink_E = 1
                        elif self.pair_cuts_explainable[frozenset([commID1,commID2])] < 0:
                            pair_mustlink_E = -1
                        else:
                            pair_mustlink_E = 0
                    else:
                        pair_mustlink_E = 0
                        
                        
                
                merge_entry = [-pair_mustlink_L,-pair_mustlink_E, -deltaH1k, frozenset([commID1,k])]
                heapq.heappush(merge_queue,merge_entry)
                merge_map[frozenset([commID1,k])] = merge_entry
            for k in self.connections[commID2]:
                self.pair_cuts[frozenset([commID1,k])] = self.pair_cuts[frozenset([commID2,k])]
                self.pair_cuts.pop(frozenset([commID2,k]))
                deltaH1k = self.merge_deltaH_SSE(commID1,k)
                merge_entry = merge_map.pop(frozenset([commID2,k]))
                merge_entry[-1] = frozenset([])
                
                pair_mustlink_L = 0
                if self.mustlink_first_L:
                    if frozenset([commID1,commID2]) in self.pair_cuts_label: #如果当前节点对在关系图中
                        if self.pair_cuts_label[frozenset([commID1,commID2])] > 0:  #在关系图中两个点的熵大于大于0  pair_mustlink = 1
                            pair_mustlink_L = 1
                        elif self.pair_cuts_label[frozenset([commID1,commID2])] < 0:
                            pair_mustlink_L = -1
                        else:
                            pair_mustlink_L = 0
                    else:
                        pair_mustlink_L = 0
            
                pair_mustlink_E = 0            
                if self.mustlink_first_E:
                    if frozenset([commID1,commID2]) in self.pair_cuts_explainable: 
                        if self.pair_cuts_explainable[frozenset([commID1,commID2])] > 0:  
                            pair_mustlink_E = 1
                        elif self.pair_cuts_explainable[frozenset([commID1,commID2])] < 0:
                            pair_mustlink_E = -1
                        else:
                            pair_mustlink_E = 0
                    else:
                        pair_mustlink_E = 0
                        
                merge_entry = [-pair_mustlink_L,-pair_mustlink_E, -deltaH1k, frozenset([commID1,k])]
          
                heapq.heappush(merge_queue, merge_entry)
                merge_map[frozenset([commID1,k])] = merge_entry
                self.connections.get(k).remove(commID2)
                self.connections.get(k).add(commID1)
                self.connections.get(commID1).add(k)
            self.connections.get(commID2).clear()
            # self.connections.pop(commID2)

    def build_tree(self):
        self.init_encoding_tree()
        self.merge()
        # y, _ = self.communities2label()
        y = self.refinement_SSE()
        return y

    def communities2label(self):
        y_pred = np.zeros(self.graph.num_nodes, dtype=int)
        label2commID = dict()
        for i, ci in enumerate(sorted(self.communities.keys())):
            y_pred[np.array(list(self.communities[ci][0])).astype(int)] = i
            label2commID[i] = ci
        return y_pred, label2commID




if __name__=='__main__':
    graph = read_graph("E:/constrained_clustering/constrainedSE/lymph6graph/Lymph6Graph")
    graph_con = Graph(graph.num_nodes)
    flatSSE = FlatSSE(graph, graph_con)
    flatSSE.build_tree()
    print(flatSSE.communities)
    print(flatSSE.SSE)
    
