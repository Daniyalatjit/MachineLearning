# Project name: Social Network Analysis
# Author: Daniyal Khan
# Date: Tue 11/03/2020 
# Time: 22:50:25.05

# In this project I will try to solve some of the question in the interconnected
# Facebook Frinedship Networks.

# We will be using NetworkX for creating graphs. NetworkX is Pyhton toolbox for 
# the creation, manipulation and study of the structure, dynamics and functiosn
# of complex networks.
import networkx as nx
import matplotlib.pyplot as plt

# Basic use of NetwrokX
# I'm creating a graph as follows:

#    C       E
#    |      / \
#    A ___ B___D
  
G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'E')
G.add_edge('B', 'D')
G.add_edge('D', 'E')

# Here the graph is printed
nx.draw_networkx(G)

# Note: for creating directed graph nx.DiGraph is used

# For this project I've cited dataset from e Stanford Large Network Dataset (SNAP)
# This dataset consists of a network representating friendship between Facebook users.
# The Facebook data has been anonymized by replacing the internal Facebook identifiers
# for each user with a new value
# Visit http://snap.stanford.edu/data/ for more information about the dataset

# let us load and extract some basic information from the graph./
fb = nx.read_adjlist("facebook_combined.txt")
fb_n, fb_k = fb.order(), fb.size()
fb_avg_deg = fb_k / fb_n
print('Nodes: ', fb_n)
print('Edges: ', fb_k)
print('Average Degree: ', int(fb_avg_deg))
degrees = [v for k, v in fb.degree() ]
degree_hist = plt.hist(degrees, 100)

# In the graph I can see that Facebook network is a scale-free network
print('Connected Componets: ', nx.number_connected_components(fb))

# As the number of components in the Facebook Network is one hence it represents a connected 
# graph
# To find communities I will prune the graph by removing some of the nodes from the graph

fb_pruned = nx.read_adjlist("facebook_combined.txt")
fb_pruned.remove_node('0')
print('Remaining Nodes: ', fb_pruned.number_of_nodes())
print('Number of Connected Components: ', nx.number_connected_components(fb_pruned))

# Here are 19 communites in the network
# Sizes of the connected components
fb_components = nx.connected_components(fb_pruned)
print('Size of the connected components: ', [len(c) for c in fb_components])


# Now the next thing I will anayze is centrality of nodes in the network.
# Centrality of a node measures its relative importance within the graph.
# The central nodes are probable more influential, have grater access to information
# , and can communicate their opinions more efficiently. By analyzing the centrality 
# of a node in social network we can determine which person is more influential, 
# most informed and most communicative.
# There are for best-known measures of centrality:
# 1-    Degree Centrality
# 2-    Betweeness Centrality
# 3-    Closness Centrality
# 4-    Eigenvector Centrality
# read more in the document....

