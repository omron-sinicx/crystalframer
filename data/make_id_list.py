import os
import pandas as pd
import numpy as np
import pickle
import os

splits = ["train", "val", "test"]
datasets = [
    "jarvis__megnet",
    "jarvis__megnet-shear",
    "jarvis__megnet-bulk",
    "jarvis__megnet-shear_modulus",
    "jarvis__megnet-bulk_modulus",
    "jarvis__dft_3d_2021",
    "jarvis__dft_3d_2021-ehull",
    "jarvis__dft_3d_2021-mbj_bandgap",
    "jarvis__oqmd_3d",
    "jarvis__oqmd_3d-bandgap",
]

check_data = {
    "jarvis__megnet": {
        "train": (60000, "mp-569112", "mvc-6917"),
        "val": (5000, "mp-558851", "mp-542332"),
        "test": (4239, "mp-554286", "mp-17481"),
    },
    "jarvis__megnet-shear": {
        "train": (4664, "mp-10000", "mp-715469"),
        "val": (392, "mp-10015", "mp-867194"),
        "test": (393, "mp-1001019", "mp-704419"),
    },
    "jarvis__megnet-bulk": {
        "train": (4664, "mp-10000", "mp-715469"),
        "val": (393, "mp-10015", "mp-867194"),
        "test": (393, "mp-1001019", "mp-704419"),
    },
    "jarvis__megnet-shear_modulus": {
        "train": (4664, "mp-10000", "mp-715469"),
        "val": (392, "mp-10015", "mp-867194"),
        "test": (393, "mp-1001019", "mp-704419"),
    },
    "jarvis__megnet-bulk_modulus": {
        "train": (4664, "mp-10000", "mp-715469"),
        "val": (393, "mp-10015", "mp-867194"),
        "test": (393, "mp-1001019", "mp-704419"),
    },
    "jarvis__dft_3d_2021": {
        "train": (44578, "JVASP-21450", "JVASP-28366"),
        "val": (5572, "JVASP-68836", "JVASP-5650"),
        "test": (5572, "JVASP-38636", "JVASP-90134"),
    },
    "jarvis__dft_3d_2021-ehull": {
        "train": (44296, "JVASP-33661", "JVASP-14073"),
        "val": (5537, "JVASP-101868", "JVASP-118132"),
        "test": (5537, "JVASP-8413", "JVASP-90134"),
    },
    "jarvis__dft_3d_2021-mbj_bandgap": {
        "train": (14537, "JVASP-11762", "JVASP-12820"),
        "val": (1817, "JVASP-59652", "JVASP-10869"),
        "test": (1817, "JVASP-100464", "JVASP-65765"),
    },
    "jarvis__oqmd_3d": {
        "train": (654108, "838046", "368371"),
        "val": (81763, "897017", "764365"),
        "test": (81763, "820496", "452980"),
    },
    "jarvis__oqmd_3d-bandgap": {
        "train": (653388, "307903", "932283"),
        "val": (81673, "719433", "989134"),
        "test": (81673, "506427", "452988"),
    },
}

def get_id(data):
    id_val = data['id'] if 'id' in data else data['material_id']
    return str(id_val)

data_dir = "./"

all_ok = True
ng_count = 0
for dataset in datasets:
    for split in splits:
        path = os.path.join(data_dir, f'{dataset}/{split}/raw/raw_data.pkl')
        if os.path.exists(path):
            crystals = pd.read_pickle(path)
            # check if the dataset size, and first and last items' IDs
            # match with the correct values
            ngt, id0gt, id1gt = check_data[dataset][split]
            n = len(crystals)
            id0 = get_id(crystals[0])
            id1 = get_id(crystals[-1])
            id_list = []
            structure_dict_lattice = {}
            structure_dict_atom_num = {}
            for i in range(n):
                id = get_id(crystals[i])
                if id in structure_dict_lattice:
                    id = get_id(crystals[i])
                    this_lattice = np.array(crystals[i]["structure"].lattice.matrix)
                    this_atom_num = len(crystals[i])
                    if np.sum(np.abs(this_lattice-structure_dict_lattice[id])) > 1e-5 or structure_dict_atom_num[id] != this_atom_num:
                        duplicate_structures = [crystals[idx] for idx in range(i) if get_id(crystals[idx]) == get_id(crystals[i])]
                        print(duplicate_structures[0])
                        print(crystals[i])
                        raise Exception("Duplicate ID")
                else:
                    structure_dict_lattice[id] = np.array(crystals[i]["structure"].lattice.matrix)
                    structure_dict_atom_num[id] = len(crystals[i])
                id_list.append(get_id(crystals[i]))

            ok = n==ngt and id0==id0gt and id1==id1gt
            ng_count += (0 if ok else 1)
            ok_str = 'OK' if ok else 'NG'

            # Show label types. Types are not checked
            # but should be all float
            types = [type(crystals[0][key]).__name__ for key in crystals[0] if key not in ['material_id', 'id', 'structure']]
            types = " ".join(types)
            print(ok_str, dataset, split, n, id0, id1, types, sep='\t')
            all_ok = ok and all_ok

            os.makedirs("splits/" + str(dataset), exist_ok=True)
            with open("splits/" + str(dataset) + "/" + str(split) + ".pkl", mode='wb') as fo:
                pickle.dump(id_list, fo)

print("ALL OK") if all_ok else print(f"{ng_count} NGs")