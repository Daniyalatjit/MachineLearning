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

# let us load and extract some basic information from the graph./
fb = nx.read_adjlist("facebook_combined.txt")
fb_n, fb_k = fb.order(), fb.size()
fb_avg_deg = fb_k / fb_n
print('Nodes: ', fb_n)
print('Edges: ', fb_k)
print('Average Degree: ', int(fb_avg_deg))
degrees = [v for k, v in fb.degree() ]
degree_hist = plt.hist(degrees, 100)
plt.plot(degree_hist[0], degree_hist[1])