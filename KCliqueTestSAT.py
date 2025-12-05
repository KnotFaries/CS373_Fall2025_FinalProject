# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 14:20:42 2025

@author: maiam
"""

from cnf import CNF
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, V, seed):
        """
        Initialize a random graph

        Parameters
        ----------
        V: int
            Number of vertices
        seed: int
            Random seed (each seed will give a different graph)
        """
        np.random.seed(seed)
        self.V = V
        self.edges = set([])
        ## Step 1: Generate a random permutation, and
        ## add all of the edges
        perm = np.random.permutation(V)
        for i in range(V):
            # Take out pair of indices in permutation
            i1 = perm[i]
            i2 = perm[(i+1)%V]
            # Add this edge (convention lower index goes first)
            i1, i2 = min(i1, i2), max(i1, i2)
            self.edges.add((i1, i2))


        ## Step 2: Add some additional edges to make the 
        ## problem harder
        # Figure out what edges are left
        edges_left = []
        for i in range(V):
            for j in range(i+1, V):
                if (i, j) not in self.edges:
                    edges_left.append((i, j))
        # Pick some edges in edges_left to add
        # Want to add enough edges to make it a little hard
        # to find the cycle, but too many so that it's trivial
        E = int(V**(5/4))
        E = min(E, V*(V-1)//2)
        print("{:.3f} %% edges".format(100*E/(V*(V-1)/2)))
        for i in np.random.permutation(len(edges_left))[0:E-V]:
            self.edges.add(edges_left[i])

    def draw(self, perm=[]):
        """
        perm: list of V indices
            Permutation certificate of the ham cycle
        """
        V = self.V
        theta = np.linspace(0, 2*np.pi, V+1)[0:V]
        x = np.cos(theta)
        y = np.sin(theta)
        plt.scatter(x, y, s=100, zorder=100)
        for i in range(V):
            plt.text(x[i]*1.1-0.02, y[i]*1.1-0.05, "{}".format(i))
        ## Draw each edge
        for (i, j) in self.edges:
            plt.plot([x[i], x[j]], [y[i], y[j]], c='k', linewidth=3)
        ## Draw the certificate permutation
        for k in range(len(perm)):
            i = perm[k]
            j = perm[(k+1)%V]
            plt.plot([x[i], x[j]], [y[i], y[j]], c='C1', linestyle='--', linewidth=1)
        if len(perm) > 0:
            chunk_size = 20
            s = ""
            for k in range(0, len(perm), chunk_size):
                for i in range(min(chunk_size, len(perm)-k)):
                    s += "{}".format(perm[k+i])
                    if k+i < len(perm)-1:
                        s += ","
                s += "\n"
            plt.title(s)
        plt.axis("off")
        
    def k_at_most(self, cnf,k):
        """

        Parameters
        ----------
        cnf : List
            DESCRIPTION.
        k : array
            The cique

        Returns
        -------
        None.

        """
        V= self.V #num verticies
       
        for i in k:
            for j in range (V):
                for l in range (V):
                    if j != l:
                        cnf.add_clause([((i,j), False), ((i,l), False)])
                
            
    def v_at_most(self, cnf, k):
        V= self.V #num verticies
       
        for j in range (V):
            for l in range (V):
                for i in range(V):
                    if i != l:
                        cnf.add_clause([((i,j), False), ((l,j), False)])
                
    def in_graph(self, cnf, k):
        V = self.V
        not_edges = set([])
        for i in range(V): 
            for j in range (V):
                if not (j, i) in self.edges:
                    not_edges.add((i,j))
                    not_edges.add((j,i))
        print(not_edges)
        
g= Graph(6, 1)
g.draw()
cnf = CNF()

print("K cique")

#g.v_at_most(cnf, [1,2,3])
g.in_graph(cnf, [1,2,3])
#print(cnf)
    