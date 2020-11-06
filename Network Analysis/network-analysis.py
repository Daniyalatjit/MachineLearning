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
from networkx.algorithms import centrality
from networkx.algorithms.centrality import betweenness
from networkx.algorithms.centrality import closeness
from networkx.algorithms.centrality import eigenvector
from networkx.algorithms.centrality import current_flow_betweenness
from networkx.generators.geometric import thresholded_random_geometric_graph
from networkx.generators.small import make_small_undirected_graph

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
# 2-    Betweenness Centrality
# 3-    Closness Centrality
# 4-    Eigenvector Centrality
# read more in the document....

# Degree Centrality
degree_cnet_fb = nx.degree_centrality(fb)
print('Facebook degree centrality: ', sorted( degree_cnet_fb.items(), 
                                    key= lambda x: x[1],
                                    reverse= True)[:10])
degree_hist = plt.hist(list(degree_cnet_fb.values()), 100)
plt.loglog(degree_hist[1][1:], degree_hist[0], 'b', marker= 'o')

# Betweenness, Closeness and Eigenvector Centrality
# Note this part may take a while to be executed depending on your systems speed.
# betweenness_fb = nx.betweenness_centrality(fb)
# closeness_fb = nx.closeness_centrality(fb)
# eigenvector_fb = nx.eigenvector_centrality(fb)
# print("Betweenness Centrality: ", sorted(betweenness_fb.items(),
#                                     key= lambda x:x[1],
#                                     reverse=True)[:10])
# print("Closeness Centrality: ", sorted(closeness_fb.items(),
#                                     key= lambda x:x[1],
#                                     reverse=True)[:10])
# print("Eigenvector Centrality: ", sorted(eigenvector_fb.items(),
#                                     key= lambda x:x[1],
#                                     reverse=True)[:10])

# If I only consider the graph nodes with more than the average degree of
# the network, I will do it by trimming the graph using degree centrality 
# value.
# I will define a function to trim the graph based of degree centrality value
# I will set the threshold to 21 connections.

def trim_by_degree_centrality(graph, degree = 0.01):
    gr = graph.copy()
    d = nx.degree_centrality(gr)
    # I have converted dict to list because dict elements can not be removed 
    # in an iterator.
    for n in list(gr.nodes()):
        if d[n] <= degree:
            gr.remove_node(n)
    return gr

degree_centrality_threshold = 21.0/(fb.order()-1.0)
print('Degree Centrality Threshold: ', degree_centrality_threshold)
fb_trimmed = trim_by_degree_centrality(fb, degree=degree_centrality_threshold)
print('Remaing Number of Nodes: ', len(fb_trimmed))

# I've reduced the graph from 4,039 to 2,226
# now I will compute the current flow betweenness cetrality in the trimmed graph.
# trimmed graph is not connected but current flow betwenness cemtrality needs connected graph.
# Here I will find the connected subgraphs in the trimmed graph.

# fb_subgraph = list(nx.connected_component_subgraphs(fb_trimmed))

# Note: connected_component_subgraphs functions has been deprecated with version 2.1, and finally removed with version 2.4.

# print("Number of Subgraphs Found: ", len(fb_subgraph))
# print("Number of Nodes in the 0th Subgraph: ", len(fb_subgraph[0]))
# betweenness = nx.betweenness_centrality(fb_subgraph[0])
# print("Trimmed FB Betweenness: ", sorted(betweenness.items(), 
#                                 key=lambda x: x[1],
#                                 reverse=True)[:10])
# current_flow = nx.current_flow_betweenness_centrality(fb_subgraph[0])
# print('Trimmed FB Subgraph Current Flow Betweenness Cetrality: ', sorted(current_flow.items(),
#                                                                 key= lambda x: x[1],
#                                                                 reverse=True)[:10])

# Now I will visulaize the centralities on Graphs for understanding and using the analysis
# there are several way to visulaize the network:
# random_layout
# spring_layout etc
# I using the spring layout for better understandability

pos_fb = nx.spring_layout(fb, iterations = 1000)
nsize=  nx.array([ v for v in degree_cnet_fb.values() ])
nsize = 500*(nsize - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos=pos_fb,
                                node_size=nsize)
edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=.1)
