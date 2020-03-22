import os
import ast
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


def main():
    data = pd.read_csv("./data/cluster_candidates.csv", sep=';', header=0, index_col=None)
    
    for index, irow in data.iterrows():
        material_id = irow['material_id']
        sg = irow['spacegroup']
        clusters = ast.literal_eval(irow['clusters'])
        cif = irow['cif']

        # obtain lattice vectors
        struct = Structure.from_str(cif, fmt="cif")
        lattice = struct.lattice.matrix

        for icluster in range(len(clusters)):
            cluster = clusters[icluster]
            coords = [np.matmul(lattice.T, x).tolist() for x in cluster]
        
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
        
            out_dir = './figures/'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            fig.savefig(out_dir+material_id+'_'+'_cluster'+str(icluster)+'.png', dpi=80)
            plt.close()


if __name__ == "__main__":
    main()


