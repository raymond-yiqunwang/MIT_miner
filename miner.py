import os
import sys
import ast
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import multiprocessing
from multiprocessing import Pool

# transition metals of interest
transition_metals = { 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                      'Co', 'Ni', 'Cu', 'Zn',
                      'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                      'Rh', 'Pd', 'Ag', 'Cd',
                      'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
                      'Pt', 'Au', 'Hg' }

# supporting ligands
ligands = { 'B', 'C', 'Si', 'Ge', 'Sn',
            'N', 'P', 'As', 'Sb', 'Bi',
            'O', 'S', 'Se', 'Te',
            'F', 'Cl', 'Br', 'I' }

def cluster_miner(data, dR):
    # output
    candidates = []
    
    for index, irow in data.iterrows():
        struct = Structure.from_str(irow['cif'], fmt="cif")
        species = set(str(x) for x in struct.species)
        if not (species & transition_metals and species & ligands):
            continue
    
        # cluter identification
        clusters = []
        visited = set()
        for idx, isite in enumerate(struct.sites):
            if idx in visited or str(isite.specie) not in transition_metals:
                continue
            visited.add(idx)
            cluster = [isite]
            def dfs(jsite):
                rj = jsite.specie.average_ionic_radius
                assert(rj > 0)
                for kdx, ksite in enumerate(struct.sites):
                    if (str(ksite.specie) in transition_metals) and kdx not in visited:
                        rk = ksite.specie.average_ionic_radius
                        assert(rk > 0)
                        if ksite.distance(jsite) <= (rj + rk + dR):
                            visited.add(kdx)
                            cluster.append(ksite)
                            dfs(ksite)
            dfs(isite)
            if len(cluster) != 4: continue

            ignore = False
            for s in range(len(cluster)-1):
                rs = cluster[s].specie.average_ionic_radius
                for t in range(s+1, len(cluster)):
                    rt = cluster[t].specie.average_ionic_radius
                    dist = cluster[t].distance(cluster[s])
                    assert(dist > 1.0)
                    if dist > rs + rt + dR:
                        ignore = True
            if not ignore: clusters.append([x.coords.tolist() for x in cluster])
        
        # is this a candidate MIT material?
        if len(clusters) > 0:
            sg = ast.literal_eval(irow['spacegroup'])
            print(irow['material_id'], irow['pretty_formula'], sg['symbol'])
            candidates.append([irow['material_id'], clusters, sg['symbol'], irow['cif']])

    return pd.DataFrame(candidates)


def parallel_computing(df_in, nworkers, dR):
    # initialize pool of workers
    pool = Pool(processes=nworkers)
    df_split = np.array_split(df_in, nworkers)
    args = [(data, dR) for data in df_split]
    df_out = pd.concat(pool.starmap(cluster_miner, args), axis=0)
    pool.close()
    pool.join()
    return df_out


def main():
    # read data
    MP_data = pd.read_csv("./data/fetch_MPdata.csv", sep=';', header=0, index_col=None)
    # number of workers for parallel processing
    nworkers = max(multiprocessing.cpu_count()-2, 1)
    print("number of workers: {}".format(nworkers))
    # maximum distance between "connected" atoms 
    dR = 2.0

    # cluster miner
    cluster_compounds = parallel_computing(MP_data, nworkers, dR)
    print("number of candidate cluster compounds: {}".format(cluster_compounds.shape[0]))
    print(cluster_compounds)

    # save output
    header = ['material_id', 'clusters', 'spacegroup', 'cif']
    cluster_compounds.to_csv("./data/cluster_candidates.csv", sep=';', \
                                header=header, index=None, columns=None)


if __name__ == "__main__":
    main()


