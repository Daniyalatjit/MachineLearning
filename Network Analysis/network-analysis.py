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
