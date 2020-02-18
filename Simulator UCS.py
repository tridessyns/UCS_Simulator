
# coding: utf-8

# # Graph Search Using UCS

# In[1]:


from sys import stderr

from queue import LifoQueue as stack
from queue import PriorityQueue as p_queue

def gc(queue):
    if not queue.empty():
        while not queue.empty():
            queue.get()


# # The adjacency list graph
# The following indicates the simple structure used for our graphs. The following shows a graph with nodes 0,1,2,3,4,5,6 and directed edges: $0\to 1$ with weight 1, $0\to 2$ with weight 2, etc. 

# In[2]:


ToyGraph = {0 : {1:1, 2:1, 4:7},
     1 : {0:1, 3:8, 6:8},
     2 : {0:1, 4:2},
     3 : {4:5, 6:2, 1:8, 5:1},
     4 : {0:7, 2:2, 5:2, 3:5},
     5 : {3:1, 4:2, 6:3},
     6 : {1:8, 5:3, 3:2}}


# # # Visualizing graphs: `showGraph`
# We can visualize our graphs using the networkx module. We need to load a few modules and convert our adjacency list graph to a networkx graph. This is done below in the following code which may be ignored. Things are set up to indicate the UCS solution ('green') some basic attempt has been made to indicte when the path overlay each other. 

# In[3]:


import networkx as nx
import pylab as plt
import pydot as pdot

from IPython.core.display import HTML, display, Image

#import pygraphviz
#from networkx.drawing.nx_agraph import graphviz_layout

def adjToNxGraph(G, digraph=True):
    """
    Converts one of our adjacency "list" representations for a graph into
    a networkx graph.
    """
    if digraph:
        Gr = nx.DiGraph()
    else:
        Gr = nx.Graph()

    for node in G:
        Gr.add_node(node)
        if G[node]:
            for adj in G[node]:
                Gr.add_edge(node, adj)
                Gr[node][adj]['weight'] = G[node][adj]
    return Gr

def showGraph(G, start, goal, paths = [], node_labels = 'default', 
              node_pos = 'neato', gsize = (14,14), save_file=None, digraph=True):
    """
    paths should be an array of which paths to show: paths = ['bfs', 'dfs', 'ucs']
    node_labels must be one of: 'default', 'none', or a list of labels to use.
    save_file must be an image file name with extension, i.e., save_file='my_graph.png'
    """
        
    fig, ax = plt.subplots(figsize=gsize)

    # Conver G into structure used in networkx
    Gr = adjToNxGraph(G, digraph=digraph)

    if node_pos is 'project_layout':
        # The project graphs have a particular structure.
        node_pos = dict(zip(Gr.nodes(),[(b, 9 - a) for a,b in Gr.nodes()]))
    else:
        node_pos = nx.nx_pydot.graphviz_layout(Gr, prog=node_pos, root=start)
        

    edge_weight=nx.get_edge_attributes(Gr,'weight')
    
    
    def path_edges(path):
        edges = list(zip(path[:-1], path[1:]))
        cost = sum([Gr[z[0]][z[1]]['weight'] for z in edges])
        if not digraph:
            edges += list(zip(path[1:], path[:-1]))
        return edges, cost
    
    # Process Paths:
    if 'ucs' in paths:
        ucost, back = ucs(G, start, goal)
        upath = getPath(back, start, goal)
        uedges, ucost = path_edges(upath)
    else:
        upath = []
        uedges = []
        

    node_col = ['orange'  if node in upath
                          else 'lightgray' for node in Gr.nodes()]

    if node_labels == 'default': 
        nodes = nx.draw_networkx_nodes(Gr, node_pos, ax = ax, node_color=node_col, node_size=400)
        nodes.set_edgecolor('k')
        nx.draw_networkx_labels(Gr, node_pos, ax = ax, font_size = 8)
    elif node_labels == 'none':
        nodes = nx.draw_networkx_nodes(Gr, node_pos, ax = ax, node_color=node_col, node_size=50)
    else:
        # labels must be a list
        nodes = nx.draw_networkx_nodes(Gr, node_pos, ax = ax, node_color=node_col, node_size=400)
        nodes.set_edgecolor('k')
        mapping = dict(zip(Gr.nodes, node_labels))
        nx.draw_networkx_labels(Gr, node_pos, labels=mapping, ax = ax, font_size = 8)
        

    edge_col = ['red' if edge in uedges else 'purple' for edge in Gr.edges()]
  
    edge_width = [3 if edge in uedges else 1 for edge in Gr.edges()]

    if digraph:
        nx.draw_networkx_edge_labels(Gr, node_pos, ax = ax, edge_color=edge_col, label_pos=0.3, edge_labels=edge_weight)
    else:
        nx.draw_networkx_edge_labels(Gr, node_pos, ax = ax, edge_color=edge_col, edge_labels=edge_weight)
    nx.draw_networkx_edges(Gr, node_pos, ax = ax, edge_color=edge_col, width=edge_width, alpha=.3)
    
    if save_file:
        plt.savefig(save_file)
    
    plt.show()
    
    result = "DFS gives a path of length {} with cost {}<br>".format(len(dpath) - 1, dcost) if 'dfs' in paths else ""
    result += "UCS gives a path of length {} with cost {}. UCS alway returns a minimal cost path.".format(len(upath) - 1, ucost) if 'ucs' in paths else ""

    display(HTML(result))  # Need display in Jupyter


# # Display initial toy graph

# In[4]:


showGraph(ToyGraph,0,4,gsize=(8,8))


# The following reconstructs the path from the back pointers.

# In[5]:


def getPath(backPointers, start, goal):
    current = goal
    s = [current]
    while current != start:
        current = backPointers[current]
        s += [current]
        
    return list(reversed(s))


# # Random graphs
# 
# The following make is easy to try out our algorithms on random graphs. There is a probability $p$ chance that the i -> j in this graph, where $0\leq p \leq 1$. A weight function can be provided to put weights on edges.

# In[6]:


from random import random as rand
from random import randint as randi

def genRandDiGraph(n, p = .5, weights = lambda i,j: 1, digraph=True):
   
    G = {}  # Initialize empty graph.
    
    for i in range(n):
        G.setdefault(i, {})
        if digraph:
            for j in range(n):
                if rand() < p and j != i:
                    # Simply choose whether or not to put
                    # a directed edge j -> i
                    G[i][j] = weights(i,j)
        else:
            for j in range(i + 1, n):
                # In case G[j] has not been initiated
                G.setdefault(j,{}) 
                if rand() < p:
                    # Simply choose whether or not to put
                    # an directed edge j -> i
                    G[i][j] = weights(i,j) 
                    G[j][i] = G[i][j]
                
    return G


# Play around with different weight functions. If you do not assign a weght function, all weights default to 1 and you can verify that UCS return shortes lenth paths, since now shortest length and minimal cost are the same. Setting weighs to -1, i.e. `weights = lambda i,j: -1` is interesting as UCS then wants to find a maximal "length" path. You can set `digraph=True` or `digraph=False` and see what the difference is.

# In[7]:


RandomG = genRandDiGraph(10, .4, weights=lambda i,j : randi(1, 15), digraph=True)
showGraph(RandomG, 4, 3, gsize=(20,20), digraph=True)


# 
# # Uniform Cost Search (UCS) 
# Here we must switch from regular queue and stack to the priority queue and introduce the cost function. Often the goal is simply to get the least cost of a path, but sometimes we wish to have the path itself so we keep track of back pointers as in the BFS/DFS so we can reconstuct the path. UCS is gauranteed to produce a path of minimal cost.

# In[8]:


def ucs(G, start, goal, trace=False):
    """
    This returns the least cost of a path from start to goal or reports
    the non-existence of such path. This also retuns a pack_pointer from
    which the search tree can be reconstructed as well as all paths explored
    including the one of interest.
    
    Usage: cost, back_pointer = ucs(Graph, start, goal)
    """
    
    # Make sure th queue is empty. (Bug in implementation?)
    fringe = p_queue()
    gc(fringe)
    
    # If we did not care about the path, only the cost we could 
    # omit this block.
    cost = {}  # If all we want to do is solve the optimization
    back_pointer = {}  # problem, neither of these are necessary.
    cost[start] = 0
    # End back_pointer/cost block
    
    current = start
    fringe.put((0, start)) # Cost of start node is 0
    closed = set()
    
    while True:
        # If the fringe becomes empty we are out of luck
        if fringe.empty():
            print("There is no path from {} to {}".format(start, goal), file=stderr)
            return None
        
        # Get the next closed element of the closed set. This is complicated
        # by the fact that our queue has no delete so items that are already
        # in the closed set might still be in the queue. We must make sure not
        # to choose such an item.
        while True:
            current_cost, current = fringe.get()  
            if current not in closed:
                # Add current to the closed set
                closed.add(current)
                if trace:
                    print("Add {} to the closed set with cost {}".format(current,current_cost))
                break
            if fringe.empty():
                print("There is no path from {} to {}".format(start, goal), file=stderr)
                return None


        # If current is the goal we are done.
        if current == goal:
            return current_cost, back_pointer
          
        # Add nodes adjacent to current to the fringe
        # provided they are not in the closed set.
        if G[current]:  # Check if G[current] != {}, bool({}) = False 
            for node in G[current]:
                if node not in closed:
                    node_cost = current_cost + G[current][node]
                    
                    # Note this little block could be removed if we only
                    # cared about the final cost and not the path
                    if node not in cost or cost[node] > node_cost:
                        back_pointer[node] = current
                        cost[node] = node_cost
                        if trace:
                            print("{current} <- {node}".format(current,node))
                    # End of back/cost block.
                    
                    fringe.put((node_cost, node))  
                    if trace:
                        print("Add {} to fringe with cost {}".format(node,node_cost))


# In[9]:


showGraph(ToyGraph, 0, 6, paths=['ucs'], gsize=(8,8))


# In[10]:


showGraph(RandomG, 4, 3, paths=['ucs'], gsize=(50,50))

