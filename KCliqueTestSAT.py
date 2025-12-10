# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 14:20:42 2025

@author: maiam
@author: russella
@author: jaheimo

Graph class:
    - Generate a random graph with V vertices and edges.
    - Reduce the k-clique problem to SAT using a CNF formula.
        - CNF variables (i, j) is True <=> "position i in the clique uses vertex j"
    - Positions i range from 0 to k-1
    - Positions j range from 0 to V-1
    - We call CNF's SAT solver

"""

from cnf import CNF
import numpy as np
import matplotlib.pyplot as plt
import random
import time


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
            Permutation certificate of the clique
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
            j = perm[(k+1)%len(perm)]
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
        
    def k_at_most(self, cnf, k):
        """
        Add "at most one vertex per clique position" constraints to the CNF
        Parameters
        ----------
        cnf : CNF
            CNF formula we're building
        k : int
            The clique size
        """
        V= self.V #num verticies
        for i in range(k):
            for j in range (V):
                for l in range (j+1, V):
                    cnf.add_clause([((i,j), False), ((i,l), False)])
                
            
    def v_at_most(self, cnf, k):
        """
        Add "per vertex appears in at most one clique position" constraints to the CNF
        Parameters
        ----------
        cnf : CNF
            CNF formula we're building
        k : int
            The clique size
        """
        V = self.V #num verticies
        for j in range (V):
            for i in range (k):
                for l in range(i+1, k):
                    cnf.add_clause([((i,j), False), ((l,j), False)])
    
    def k_at_least(self, cnf, k):
        """
        Add "at least one vertex per clique position" constraints to the CNF
        Parameters
        ----------
        cnf : CNF
            CNF formula we're building
        k : int
            The clique size
        """
        V = self.V #num verticies
        for i in range(k):
            clause = []
            for j in range (V):
                clause.append(((i, j), True))
            cnf.add_clause(clause)

    def in_graph(self, cnf, k):
        V = self.V
        not_edges = set([])
        for i in range(V): 
            for j in range (i+1, V):
                if (i, j) not in self.edges and (j,i) not in self.edges:
                    not_edges.add((i,j))
        for m in range (k):
            for l in range (m+1, k):
                for (i, j) in not_edges:
                    cnf.add_clause([((m, i), False), ((l, j), False)])
                    cnf.add_clause([((m, j), False), ((l, i), False)])
        
    def get_cnf_formula(self, k):
        """
        Do a reduction from this problem to SAT by filling in 
        CNF formulas

        Returns
        -------
        CNF: CNF Formula corresponding to the reduction
        """
        cnf = CNF()
        self.k_at_most(cnf, k)
        self.v_at_most(cnf, k)
        self.k_at_least(cnf, k)
        self.in_graph(cnf, k)
        return cnf
    
    def solve(self, k):
        cnf = self.get_cnf_formula(k)
        cert = cnf.solve_glucose()

        if len(cert) == 0:
            return []
        
        # Translate SAT solution back to my language
        pos_to_vertex = {}
        for (i, j), val in cert.items():
            if val:
                if i not in pos_to_vertex:
                    pos_to_vertex[i] = j
        clique = [pos_to_vertex[i] for i in sorted(pos_to_vertex.keys())]
        return clique
    
    def check_cert(self, clique, k):
        is_valid = True
        ## Step 1: Check that it's k size
        if len(clique) != k:
            is_valid = False
        ## Step 2: Check edges between all pairs
        for i in range(k):
            for j in range(i+1, k):
                c_i, c_j = clique[i], clique[j]
                is_valid = is_valid and ((c_i, c_j) in self.edges or (c_j, c_i) in self.edges)
        return is_valid


def search_hard_instances(num_trials=100):
    hardest_time = -1
    hardest_info = None

    for trial in range(num_trials):
            # Randomly sample graph size
            V = random.randint(10, 22)
            # Randomly sample clique size
            k = random.randint(3, min(6, V - 1))
            # Random seed
            seed = random.randint(0, 10)
            # Enforce the rule: variables = V * k <= 200
            if V * k > 200:
                continue
            g = Graph(V, seed)
            cnf = g.get_cnf_formula(k)
            num_vars = len(cnf.vars)
            num_clauses = len(cnf.clauses)
            if num_vars > 200:
                continue
            t0 = time.time()
            cert = cnf.solve_glucose()
            t1 = time.time()
            runtime = t1 - t0

            sat = (len(cert) > 0)

            if sat and runtime > hardest_time:
                hardest_time = runtime
                hardest_info = {
                    "V": V,
                    "k": k,
                    "seed": seed,
                    "time": runtime,
                    "sat": sat,
                    "num_vars": num_vars,
                    "num_clauses": num_clauses,
                    "cnf": cnf
                }
    if not hardest_info:
        print("No valid instances found.")
        return
    # Save the hardest found instance
    filename = "hard_kclique_instance_random.cnf"
    hardest_info["cnf"].save(filename)

    print("\n\nHARD INSTANCE FOUND")
    print(f"Saved as: {filename}")
    print(f"V = {hardest_info['V']}")
    print(f"k = {hardest_info['k']}")
    print(f"seed = {hardest_info['seed']}")
    print(f"sat = {hardest_info['sat']}")
    print(f"num_vars = {hardest_info['num_vars']}")
    print(f"num_clauses = {hardest_info['num_clauses']}")
    print(f"runtime = {hardest_info['time']:.3f}s")

## Test Parts 2 and 3
num_V = 19
k = 3
seed = 10
g = Graph(num_V, seed)
clique = g.solve(k)
if not clique:
    print(f"{k}-clique not found.")
else:
    print(f"Clique: {clique}")
    print(f"Valid: {g.check_cert(clique, k)}")
    g.draw(clique)
    plt.show()

## Search for hard instance for Part 4
# Hardest run so far:
# V = 19
# k = 3
# seed = 10
# search_hard_instances()