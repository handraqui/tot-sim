
# coding: utf-8

# In[1]:


from random import randint


# In[2]:


def Contains(A,B):

    for b in B:
        
        if(b not in A):

            return False

    return True


# In[3]:


def Heirs(vertex,graph):
    
    N = len(graph)

    heirs = [j for j in range(N) if graph[vertex][j] == 1]

    return heirs


# In[4]:


def Multi_Heirs(vertices, graph):

    multi_heirs = [Heirs(i,graph) for i in vertices]

    multi_heirs = [n for sub in multi_heirs for n in sub]

    return list(set(multi_heirs))


# In[5]:


def Parents(vertex, graph):
    
    N = len (graph)
    
    parents = [j for j in range(N) if graph[j][vertex] == 1]
    
    return parents
    


# In[6]:


def Multi_Parents(vertices, graph):

    multi_parents = [Parents(i,graph) for i in vertices]

    multi_parents = [n for sub in multi_parents for n in sub]

    return list(set(multi_parents))


# In[7]:


def Ancestors(vertex, graph):
    
    ancestors = []
    
    new_ancestors = Parents(vertex, graph)
    
    while (not Contains(ancestors, new_ancestors)):
        
        ancestors += new_ancestors
        ancestors = list(set(ancestors))
        
        new_ancestors += Multi_Parents(new_ancestors, graph)
        new_ancestors = list(set(new_ancestors))
        dag
    return ancestors


# In[181]:


def Multi_Ancestors(vertices,graph):
    
    multi_ancestors = [Ancestors(i,graph) for i in vertices]
    
    multi_ancestors = [n for sub in multi_ancestors for n in sub]
    
    return list(set(multi_ancestors))


# In[8]:


def Descendants(vertex, graph):

    descendants = []

    new_descendants = Heirs(vertex, graph)

    while(not Contains(descendats, new_descendants)):

        descendants += new_descendants
        descendants = list(set(descendants))

        new_descendants += Multi_Heirs(new_descendants, graph)
        new_descendants = list(set(new_descendants))


    return descendants


# In[180]:


def Multi_Descendants(vertices,graph):
    
    multi_descendants = [Descendants(i,graph) for i in vertices]
    
    multi_descendants = [n for sub in multi_Descendants for n in sub]
    
    return list(set(multi_descendants))


# In[9]:


def Has_Cycle(graph):
    
    N = len(graph)

    for i in range(N):
        
        if(i in Descendants(i,graph)):

            return True
        
    return False


# In[124]:


def Which_Cycle(graph):
    
    N = len(graph)

    for i in range(N):
        
        if(i in Descendants(i,graph)):

            print(i)

if Has_Cycle(dag):
    print("Olha aqui:")
    Which_Cycle(dag)


# In[136]:


def PrintAncestors(vertex,graph):
    if len(Parents(vertex,graph)) != 0:
        print (str(vertex),": ", Parents(vertex,graph))
        for i in Parents(vertex,graph):
            PrintAncestors(i,graph)
            
def PrintDescendants(vertex,graph):
    if len(Heirs(vertex,graph)) != 0:
        print (str(vertex),": ", Heirs(vertex,graph))
        for i in Heirs(vertex,graph):
            PrintDescendants(i,graph)


# In[179]:


# Classificação de vértices

def FirstGeneration(graph):
    return [j for j in range(len(graph)) if len(Ancestors(j,graph)) == 0]
    
def LastGeneration(graph):
    return [j for j in range(len(graph)) if len(Descendants(j,graph)) == 0]

def Isolated(graph):
    return [j for j in range(len(graph)) if (len(Parents(j,graph)) == 0 and len(Heirs(j,graph)) == 0)]


# In[211]:


# Topologia do grafo: Retorna todos os vértices com algum caminho de parentesco

def Connected_Subgraph (vertex, graph):
    
    connected = []
    
    new_connected = Ancestors(vertex,graph) + Descendants(vertex,graph)
    
    while not Contains(connected,new_connected):
        
        connected += new_connected
        connected = list(set(connected))
        
        new_connected += Multi_Ancestors(connected,graph) + Multi_Descendants(connected,graph)
        new_connected = list(set(new_connected))
        
    return connected


# In[11]:


def RanDAG(n_vertices, n_edges):
    
    assert( n_edges <= (n_vertices-1)*(n_vertices)/2)

    ranDag = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]

    edge_count = 0

    while(edge_count < n_edges):
        
        i = randint(0,n_vertices-1)
        j = randint(0,n_vertices-1)

        if( ranDag[i][j] == 0 ):

            ranDag[i][j] = 1

            if( Has_Cycle(ranDag) ):

                ranDag[i][j] = 0
            
            else:

                edge_count += 1

    return ranDag      


# In[85]:


def RanDAG2 (n_vertices, n_edges):
    
    assert( n_edges <= (n_vertices-1)*(n_vertices)/2)

    ranDag = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]

    edge_count = 0
    
    multi_ancestors = [[i] for i in range(n_vertices)]
    
    counts = 0
    
    possible_parents = [j for j in range(n_vertices)]
    
    while edge_count < n_edges:
        
        if len(possible_parents) == 0:
            
            print("Não há mais arestas possíveis")
            
            break
        
        i = possible_parents[randint(0,len(possible_parents)-1)]
        
        available = []
        
        for j in range(n_vertices):
            
            if j not in multi_ancestors[i]:
                
                available.append(j)
        
        j = available[randint(0,len(available)-1)]
        
        ranDag[i][j] = 1
        
        edge_count+= 1
        
        for l in range(n_vertices):
            
            multi_ancestors[l] = multi_ancestors[l] + [k for k in multi_ancestors[i]]
            
            multi_ancestors[l] = list(set(multi_ancestors[l]))
            
        for l in possible_parents:
            
            if len(multi_ancestors[l]) == n_vertices:
                
                possible_parents.remove(l)
        
        if counts == 100000:
            
            print("Olha, demorou demais: ",edge_count," iterações. Pediu muitas arestas? Mexa no script e aumente o limite.")
            
            break
            
        counts += 1
    
    return ranDag


# In[307]:


# Triangular inferior, n_edges binomial

from numpy.random import binomial

def RanDAG3(n_vertices,n_edges):
    
    max_possible = (n_vertices-1)*(n_vertices)/2
    
    p = n_edges/((n_vertices-1)*(n_vertices)/2)

    graph = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]
    
    for i in range(len(graph)):
        
        for j in range(len(graph)):
            
            if j < i :
                
                graph[i][j] = int(randint(0,(n_vertices-1)*(n_vertices)/2) <= n_edges)

    return graph


# In[208]:


get_ipython().run_cell_magic('time', '', 'dag = RanDAG(80,3000)\nprint(Has_Cycle(dag))')


# In[416]:


get_ipython().run_cell_magic('time', '', 'dag = RanDAG2(80,3000)\nprint(Has_Cycle(dag))')


# In[483]:


get_ipython().run_cell_magic('time', '', 'dag = RanDAG3(80,3000)\nprint(Has_Cycle(dag))\n\nimport numpy as np\nprint("Quantidade de arestas: ",np.sum(np.array(dag)))')


# In[212]:


def Distribute(graph):

    N = len(graph)

    distributed_graph = [[0 for _ in range(N)] for _ in range(N)]

    for i in range(N):

        total_weight = sum(graph[i])

        for j in range(N):

            if( graph[i][j] == 1 ):

                distributed_graph[i][j] = 1 / total_weight

    return distributed_graph


# In[499]:


class DAG():
    
    def __init__(self,n_vertices,n_edges,generator=1):
        assert( generator in [1,2,3])
        if generator == 1:
            self.matrix = RanDAG(n_vertices,n_edges)
        if generator == 2:
            self.matrix = RanDAG2(n_vertices,n_edges)
        if generator == 3:
            self.matrix = RanDAG3(n_vertices,n_edges)
        
        self.n_vertices = n_vertices
        self.n_topics = n_vertices
        self.n_edges = n_edges
        if generator == 3:
            self.n_edges = sum([k for sub in self.matrix for k in sub])
        self.prob_matrix = Distribute(self.matrix)
    
    def Descendants(self,vertex):
        return Descendants(vertex,self.matrix)
        
    def Ancestors(self,vertex):
        return Ancestors(vertex,self.matrix)
    
    def Heirs(self,vertex):
        return Heirs(vertex,self.matrix)
    
    def Parents(self,vertex):
        return Parents(vertex,self.matrix)

    def Multi_Descendants(self,vertex):
        return Multi_Descendants(vertex,self.matrix)
        
    def Multi_Ancestors(self,vertex):
        return Multi_Ancestors(vertex,self.matrix)
    
    def Multi_Heirs(self,vertex):
        return Multi_Heirs(vertex,self.matrix)
    
    def Multi_Parents(self,vertex):
        return Multi_Parents(vertex,self.matrix)
    
    def PrintAncestors(self,vertex):
        PrintAncestors(vertex,self.matrix)
    
    def PrintDescendants(self,vertex):
        PrintDescendants(vertex,self.matrix)
    
    def __str__(self):
        string = ""
        for i in range(self.n_vertices):
            string += str(self.matrix[i])
            string += "\n"
        return str(string)


# In[507]:


a = DAG(20,180,3)
print(a)
print(a.n_edges)

