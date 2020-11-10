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
from networkx.algorithms.centrality.closeness import closeness_centrality
from networkx.generators.geometric import thresholded_random_geometric_graph
from networkx.generators.small import make_small_undirected_graph
import numpy as np

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
betweenness_fb = nx.betweenness_centrality(fb)
closeness_fb = nx.closeness_centrality(fb)
eigenvector_fb = nx.eigenvector_centrality(fb)
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
nsize=  np.array([ v for v in degree_cnet_fb.values() ])
nsize = 500*(nsize - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos=pos_fb,
                                node_size=nsize)
edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=.1)

# Changing the centrality measure to closeness centrality and eigenvector centrality, I found
# different spring_layout graph but mostly similar. 
nsize=  np.array([ v for v in closeness_centrality.values() ])
nsize = 500*(nsize - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos=pos_fb,
                                node_size=nsize)
edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=.1)

nsize=  np.array([ v for v in eigenvector_fb.values() ])
nsize = 500*(nsize - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos=pos_fb,
                                node_size=nsize)
edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=.1)

nsize=  np.array([ v for v in closeness_fb.values() ])
nsize = 500*(nsize - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos=pos_fb,
                                node_size=nsize)
edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=.1)

# **Page Rank**
# PageRank is an Algorithm invented by Larry Page and Sergey Brin, and bacame a Google trademark in 1998.
# PageRank Algorithm is used to rate pages objectively and effectively.
# If we consider a node as a webpage then then hyperlink to the page counts as a vote of support and a page has high rank. If the sum of the ranks of its incoming edges is high then the PageRank is high.

# The PageRank algorithm is described froma probabilistic point of view. 
# If we consider that one node has n edges to n nodes then we can say that the probabiltiy of going to anyone of the page is 1/n (initially).
# Then we calculate the probablity again using  the formula: PR = Sum(Old PR of incomming i(th) node/outgoing edges of incoming i(th) node)
# After few iteration, the page having higher probablity value get higher rank.

pr = nx.pagerank(fb, alpha = 0.85)
nsize = np.array([ v for v in pr.values() ])
nsize = 500*(nsize - min(nsize)) / (max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb,
                               pos = pos_fb,
                               node_size = nsize)
edges = nx.draw_networkx_edges(fb,
                               pos = pos_fb,
                               alpha = 0.1)

# Ego Networks
# Ego networks are subnetworks of neighbors that are centered on a certain node. In Facebook and Linkedin
# these are described as "your network". Every node in a network has its own ego network and can only access
# the ndeos in it. All ego-networks interlock to form the whole social network.

ego_107 = nx.ego_graph(fb, '107')
print('Number of nodes in ego graph 107: ', len(ego_107))
print('Number of nodes in ego graph having radious up to 2: ', 
      len(nx.ego_graph(fb, '107', radius=2 )))

# The ego network of node 107 has 1,046 nodes while when I expand its radius upto 2 then it can reach more nodes
# (2,687), which is a quite lasrge network having half of the nodes of whole network. 

# The dataset we are using in this notebook provides the previously labeled ego-networks, I will compute the 
# actual size of the ego-network following the user labeling.

ego_id = 107
net_107 = nx.read_edgelist(
    os.path.join('/content/drive/My Drive/Machine Learning/facebook', '{0}.edges'.format(ego_id)),
    nodetype = int
)

print('Nodes of the ego graph 107: ', len(net_107))

# I will try to understand the structure of the Facebook Network by comparing the 10 different ego-networks 
# among them.
# I will compute the number of edges in every ego network then will compare and choose the most dense ego network.

from numpy import zeros
ego_ids = (0, 107, 348, 414, 686, 698, 1684, 3980, 1912, 3437)
ego_sizes = zeros((10,1))
i = 0

for id in ego_ids:
  grp = nx.read_edgelist(
      os.path.join('/content/drive/My Drive/Machine Learning/facebook', '{0}.edges'.format(id)),
      nodetype = int
  )
  ego_sizes[i] = grp.size()
  i=i+1
[i_max, j] = (ego_sizes == ego_sizes.max()).nonzero()
ego_max = np.array(ego_ids)[i_max]
print('The most dense ego network is: ', ego_max[0])
G = nx.read_edgelist(
    os.path.join('/content/drive/My Drive/Machine Learning/facebook', '{0}.edges'.format(ego_max[0])),
    nodetype = int
)

print('Nodes: ', G.order())
print('Edges: ', G.size())
print('Average Degree: ', int(G.size()/G.order()))

# Now I will compute that how many intesections exists between the ego-network in the Facebook Network. To do this, I 
# will add a field ego_net for every node and store an array with ego-networks the node belongs to. Then having length 
# of these arrays, I will be able to compute the number of nodes that belongs to 1, 2, 3, 4 amd more than 4 ego-networks.

for i in fb.nodes() :
  fb.nodes[str(i)]['egonet'] = []

for id in ego_ids:
  G = nx.read_edgelist(
      os.path.join('/content/drive/My Drive/Machine Learning/facebook', 
                   '{0}.edges'.format(id)),
                   nodetype = int
  )
  print(id)
  for n in G.nodes() :
    if (fb.nodes[str(n)]['egonet'] == []) :
      fb.nodes[str(n)]['egonet'] = [id]
    else :
      fb.nodes[str(n)]['egonet'].append(id)

# Computing the intersections
intersects = [len(x['egonet']) for x in fb.nodes.values() ]
print('Number of node into 0 ego-network:', sum(np.equal(intersects, 0)))
print('Number of node into 1 ego-network:', sum(np.equal(intersects, 1)))
print('Number of node into 2 ego-network:', sum(np.equal(intersects, 2)))
print('Number of node into 3 ego-network:', sum(np.equal(intersects, 3)))
print('Number of node into 4 ego-network:', sum(np.equal(intersects, 4)))
print('Number of nodes into more than 4 ego-network:', sum(np.greater(intersects, 4)))

# Now, I will visualize the different ego-networks using different colors so that different communities in the 
# network could be identified.

for i in fb.nodes():
  fb.nodes[str(i)]['egocolor'] = 0

idColor = 1
for id in ego_ids:
  G = nx.read_edgelist(
      os.path.join('/content/drive/My Drive/Machine Learning/facebook', 
                   '{0}.edges'.format(id)),
                   nodetype = int)
  for n in G.nodes():
    fb.nodes[str(n)]['egocolor'] = idColor
    idColor += 1
  
  colors = [x['egocolor'] for x in fb.nodes.values() ]
  nsize = np.array([v for v in degree_cent_fb.values() ])
  nsize = 500*(nsize - min(nsize))/(max(nsize) - min(nsize))
  nodes = nx.draw_networkx_nodes(
      fb, pos = pos_fb,
      cmap = plt.get_cmap('Paired'),
      node_color = colors,
      node_size = nsize
  )
  edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=.1)

# Community Detection
# A community in a network is the set of nodes of the network that is densely connected internally.
# I will use Community toolbox for implementing Louvain Method for community detection. 

import community
partition = community.best_partition(fb)
print('Number of communities found: ', max(partition.values()))
colors2 = [partition.get(node) for node in fb.nodes()]
nsize = np.array([v for v in degree_cent_fb.values() ])
nsize = 500*(nsize - min(nsize))/ (max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb,
                               pos = pos_fb,
                               cmap = plt.get_cmap('Paired'),
                               node_color = colors2,
                               node_size = nsize)
edges = nx.draw_networkx_edges(fb, pos=pos_fb, alpha=0.1)

# Conclusion
# In this notebook I used Python toolbox, NetworkX which a useful tool for network anlysis. I'm intoduced 
# to some of the basic concepts in social network analysis susch as, Centrality Measures which identifies 
# the importance of a node in the network or community or ego-network, allows to study the reach of the 
# information a node can transmit or have access to. 

# I tried to resolve several issues, such as finding the most representative members of the network in 
# terms of the most "connected", the most "circulated" and the "closest" or the most "accessible" nodes to 
# the others. 