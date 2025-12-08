import subprocess

# Define the command to install networkx using pip
command = 'pip install networkx'

# Execute the command using subprocess
subprocess.run(command, shell=True)
# Define the command to install GraphRicciCurvature using pip
command = 'pip install GraphRicciCurvature'

# Execute the command using subprocess
subprocess.run(command, shell=True)



import pandas as pd
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from statistics import mean

# import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import csv

import time

#read data from csv
data=pd.read_csv('data_clean_interpolated.csv',index_col='date')

#log return series
df=np.log(data/data.shift(1))

#function to convert correlation to distance
def corr_to_dist(c):
    return np.sqrt(2*(1-c))



#curvature computations
# T=22
# t=2
c_lim=0.85
d_lim=corr_to_dist(c_lim)
a=0.5

timewindows=[11,22,66,132,250]
for T in timewindows:
    N=len(df)-T
    curv=[]
    for t in range(N):
    # for t in range(10):
        # computing distances
        weights=corr_to_dist(df.iloc[t:t+T].corr())
        weights.fillna(0, inplace=True)

        #create weighted graph
        keys=df.keys()
        G=nx.Graph()
        G.add_nodes_from(keys)
        G.add_weighted_edges_from([(u,v,weights[u][v]) for u in keys for v in keys if u!=v])
        # len(G.edges())

        #minimum spanning tree of G
        H=nx.minimum_spanning_tree(G)

        # #add edges of high correlation
        H.add_weighted_edges_from([(u,v,weights[u][v]) for u in keys for v in keys if (u!=v and (u,v) not in H.edges() and weights[u][v]<=d_lim)])


        # # Plot the graph
        # pos = nx.spring_layout(H)
        # nx.draw(H, pos, with_labels=True, node_color='lightblue')
        # plt.title("MST")
        # plt.show()


        #Ricci curvature
        orc = OllivierRicci(H, alpha=a, verbose="TRACE")
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy()  # save an intermediate result
        curvatures = nx.get_edge_attributes(G_orc, "ricciCurvature").values()

        #average
        curv.append(list(curvatures))

        if t%100==0:
            print(f'T={T} {round(t/len(df)*100,2)}% complete')



    with open(f'curvatures_{T}days.pkl', 'wb') as f:
        pickle.dump(curv, f)

print('All calculations done')


