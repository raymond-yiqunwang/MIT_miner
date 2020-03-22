import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(filename, outputpath):
    data = pd.read_csv(filename, sep=';', header=0, index_col=None)
    
    for index, irow in data.iterrows():
        # obtain lattice vectors
        iCif = irow['cif']
        struct = Structure.from_str(iCif, fmt="cif")
        lattice = struct.lattice.matrix
        
        # parse string to get cluster coord.
        coords = []
        icluster = irow['cluster_coord']
        start = 1
        while True:
            while icluster[start] != '[':
                start += 1
                if start >= len(icluster): break
            if start >= len(icluster): break
            end = start + 1
            while icluster[end] != ']':
                end += 1
            icoord = icluster[start+1:end].split(',')
            icoord = [float(x) for x in icoord]
            coords.append(list(np.matmul(lattice.T, icoord)))
            start = end + 1
        
        vertices = [
            [0.0, 0.0, 0.0],
            list(lattice[0]),
            list(lattice[1]),
            list(lattice[2]),
            list(lattice[0]+lattice[1]),
            list(lattice[0]+lattice[2]),
            list(lattice[1]+lattice[2]),
            list(lattice[0]+lattice[1]+lattice[2])
        ]

        edges = [
            [vertices[0], vertices[1], vertices[4], vertices[2]],
            [vertices[0], vertices[2], vertices[6], vertices[3]],
            [vertices[0], vertices[1], vertices[5], vertices[3]],
            [vertices[3], vertices[5], vertices[7], vertices[6]],
            [vertices[2], vertices[4], vertices[7], vertices[6]],
            [vertices[1], vertices[4], vertices[7], vertices[5]],
        ]
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='w')
        faces.set_facecolor((0,0,1,0.1))

        ax.add_collection3d(faces)

        xs = []
        ys = []
        zs = []
        # add cluster
        for atom in coords:
            xs.append(atom[0])
            ys.append(atom[1])
            zs.append(atom[2])
    
        ax.scatter(xs, ys, zs, s=200, c='r', marker='o')

        hlim = max([max(x) for x in vertices])
        llim = min([min(x) for x in vertices])
        ax.set_xlim(llim, hlim)
        ax.set_ylim(llim, hlim)
        ax.set_zlim(llim, hlim)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        fig.savefig(outputpath+irow['cluster_id']+".png", dpi=80)


if __name__ == "__main__":
    dR = "dR0d9"
    filename = "./data/tetrahedron_"+dR+".csv"
    outputpath = "./figures/"+dR+"/"
    main(filename, outputpath)


