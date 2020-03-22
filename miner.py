import os
import sys
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
                      'Pt', 'Au', 'Hg'}


def cluster_miner(data, dR):
    # output
    candidates = []
    
    for index, irow in data.iterrows():
        struct = Structure.from_str(irow['cif'], fmt="cif")
        if not set(str(x) for x in struct.species) & transition_metals:
            continue
    
        # cluter identification
        clusters = []
        visited = set()
        for idx, isite in enumerate(struct.sites):
            if idx in visited or str(isite.specie) not in transition_metals:
                continue
            cluster = [isite]
            def dfs(jsite):
                rj = jsite.specie.average_ionic_radius
                assert(rj > 0)
                for jdx, ksite in enumerate(struct.sites):
                    if (str(ksite.specie) in transition_metals) and jdx not in visited:
                        rk = ksite.specie.average_ionic_radius
                        assert(rk > 0)
                        if ksite.distance(jsite) <= (rj + rk + dR):
                            visited.add(jdx)
                            cluster.append(ksite)
                            dfs(ksite)
            dfs(isite)
            if len(cluster) == 4:
                clusters.append(cluster)
        
        # is this a candidate MIT material?
        if len(clusters) > 0:
            print('one hit')
            candidates.append([irow['material_id'], clusters])

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
    dR = 1.5
    # cluster miner
    cluster_compounds = parallel_computing(MP_data, nworkers, dR)
    print(cluster_compounds.shape[0])


if __name__ == "__main__":
    main()


