#!/usr/bin/env python
# coding: utf-8

# ## algorithm design and anlysis-2025 spring  homework 4
# **Deadline**：2025.5.14
# 
# **name**:
# 
# 
# note：
# ---
# 1. 带有\*的题目，申请免上课的同学，必须完成，其他同学选作；
# 2. 请独立完成，如求助了他人或者大模型，请著明，并且不可省略算法分析部分；
# 4. 如若作答有雷同，全部取消成绩；
# 3. 需要书面作答的题目，可以通过引用图片的形式添加，但是注意上传项目时包含所引用的图片的源文件；
# 4. $log_n$ 默认表示$log_2{n}$;

# ## 问题 1 
# **最小生成树（Minimum Spanning Tree）**
# 
# 设  **G**  为一个带权重的连通无向图，且所有边的权重均不相等。令$e_i$ 为权重第 $i$ 小的边。最小生成树（MST）是否必须包含 $e_1$ ? 同理，是否必须包含 $e_2$ 和 $e_3$ ? 若必须包含，请给出证明；否则，请构造反例。需从基本原理论证，不能依赖割引理(cut lemma) 或 Prim/Kruskal算法的正确性。
# 

# answer:

# ## 问题 2 
# **瓶颈生成树（Bottleneck Spanning Tree）**
# 
# 带有权重的无向图 $G(V,E,w)$ 的瓶颈生成树，表现为：在所有生成树中，最大权重边的权重值最小。即，BST $T$ 最小化瓶颈损失 $c(T)=max_{e \in T}{w(e)}$。
# 
# 1. 证明 $G$ 的每一个最小生成树（MST）都是瓶颈生成树（BST）
# 2. 设计一个线性时间复杂度的算法：， 对于一个图 $G(V,E,w)$ 和一个整数 $b$，判断图 $ G$ 是否存在一个瓶颈生成树，其最大权重边的权重不超过 $b$，分析算法设计思路，并基于python编程实现。
# 3. 设计一个线性时间复杂度的算法：对于给定的图 $G(V,E,w)$，找到其瓶颈生成树，分析算法设计思路，并基于python编程实现。

# idea：

# In[ ]:


# 2.
# add your code here
import collections

def has_bst_with_max_weight(graph, V, b):
#     判断图 G 是否存在一个 BST，其最大权重边不超过 b。
#     graph: 字典表示的图，graph[u] = [(v, w), ...] 表示从 u 到 v 的边权重为 w
#     V: 顶点数
#     b: 最大权重阈值
    # 构建子图 ，只包含权重 <= b 的边
    G_prime = collections.defaultdict(list)
    for u in graph:
        for v, w in graph[u]:
            if w <= b:
                G_prime[u].append(v)
                G_prime[v].append(u)  # 无向图，添加反向边

    # 使用 DFS 检查连通性
    visited = [False] * V
    stack = [0]  
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in G_prime[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    return all(visited)
# 思路：瓶颈生成树（BST）的最优性在于最小化生成树中最大权重边的权重，要判断是否存在最大权重边<=b的 BST，可通过构造仅含权重<=b的子图G'
# 并用 DFS 检查其连通性;若G'连通，则存在满足条件的 BST，否则不存在。
# algorithm of the liear time complexity ：构建子图，遍历了所有的边，时间为O(E)；使用了DFS，时间为O(V+E)。因此总的时间复杂度为O(V+E)

# 3.
# add your code here
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        elif self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True

def find_bst(graph, V):
#     找到图 G 的一个瓶颈生成树。
#     返回: (瓶颈值, BST 的边列表)


    edges = []
    for u in graph:
        for v, w in graph[u]:
            if u < v:  # 避免重复添加无向边
                edges.append((w, u, v))
    edges.sort()  # 按权重升序排序

    uf = UnionFind(V)
    bst_edges = []
    bottleneck = 0

    for w, u, v in edges:
        if uf.union(u, v):
            bst_edges.append((u, v, w))
            bottleneck = w
            if len(bst_edges) == V - 1:  
                break

    return bottleneck, bst_edges
# 思路：BST 的最大权重边 c(T) 是所有生成树中可能的最大权重边的最小值，可通过类 Kruskal 算法按权重升序排序边并用并查集逐步添加直至图连通，
# 确定瓶颈值为最后添加边的权重，再在相应子图上构造生成树。
# algorithm of the liear time complexity ：排序边耗时 O(ElogE)，并查集操作耗时 O(Eα(V))，因此总的时间复杂度为O(ElogE)


# ## 问题 3
# 
# **道路网（Road Network）**
# 
# 假设有一个以图 $ G(V, E, l) $ 表示的道路网络，连接了一组城市 $ V $。我们假设该网络是有向的，并且每条道路 $(u, v) \in E$ 都有一个非负的长度 $ l(u, v) $。一条新的道路即将被建造，因此有一个列表 $ E' $ 包含它可以连接的城市对。每对 $(u, v) \in E'$ 都有一个对应的长度 $ l'(u, v) $。我们希望选择一对城市，使得两个城市 $ s, t \in V $ 之间的距离减少最大。请为此问题编写一个高效的算法，并详细解释算法的正确性和复杂度。
# 

# ## 问题 4
# 
# **逃离问题**
# 
# 一个 $ n \times n $ 的网格是一个无向图，由 $ n $ 行和 $ n $ 列的顶点组成，如下图所示。我们用 $(i,j)$ 表示第 $ i $ 行和第 $ j $ 列的顶点。除了边界顶点，网格中的所有顶点都有四个邻居，即满足 $ i = 1, i = n, j = 1 $ 或 $ j = n $ 的点 $(i,j)$。
# 
# 给定网格中的 $ m \leq n^2 $ 个起点 $(x_1, y_1), (x_2, y_2), \cdots , (x_m, y_m)$，逃离问题是确定是否存在 $ m $ 条顶点不相交的路径（即路径之间不相交），从这些起点到边界上的任意 $ m $ 个不同点。例如，图1中的网格存在逃离。
# 
# (1) 该问题可以看作是一个最大流问题。考虑一个流网络，其中顶点和边都有容量。也就是说，进入任何给定顶点的总正流量受到容量限制。证明在具有边和顶点容量的网络中确定最大流可以简化为在具有可比大小的普通流网络上的最大流问题。更准确地说，你需要将一个具有顶点和边容量的网络 $ G = (V,E) $ 转换为另一个仅具有边容量的网络 $ G' = (V', E') $，使得两个网络上的最大流相同，并且你构建的新网络具有 $ V' = O(V) $ 个顶点和 $ E' = O(E) $ 条边。你可以假设网络是连通的。
# 
# (2) 描述一个解决逃离问题的高效算法，并分析其运行时间。
# 
# 
# <div align="center"> <img alt="图片" src="./fig/escepe-p.png"> </div>
# <center> 图2. 逃脱问题网格，起始顶点为黑色，其他网格顶点为白色</center>

# idea：
