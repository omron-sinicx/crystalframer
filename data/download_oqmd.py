import jarvis
import os
import pathlib
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms

from tqdm import tqdm
import random
import numpy
import pickle
import math
import pymatgen.core

def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(numpy.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test

def clean_species(species):
    return [ s.strip() for s in species]

if __name__ == '__main__':
    # If SSL certification error occurs Under proxy, the following workaround may work.
    # os.environ['REQUESTS_CA_BUNDLE'] = ''
    # os.environ['CURL_CA_BUNDLE'] = ''

    # print(jarvis.__path__)
    # cached_files = pathlib.Path(jarvis.__path__[0]).glob("db/*.zip")
    # for file in cached_files:
    #     print(f"Removing {file.absolute()}")
    #     os.remove(file.absolute())

    datasets = [
        "oqmd_3d_no_cfid",
        "oqmd_3d_no_cfid",
    ]
    save_names = [
        "jarvis__oqmd_3d",
        "jarvis__oqmd_3d-bandgap",
    ]
    used_vals = [
        {
            '_oqmd_entry_id': ('id', str),
            '_oqmd_delta_e': ('delta_e', float),
            '_oqmd_stability': ('stability', float),
            'structure': ('structure', object)
        },
        {
            '_oqmd_entry_id': ('id', str),
            '_oqmd_band_gap': ('bandgap', float),
            'structure': ('structure', object)
        },
    ]

    for i, t in enumerate(datasets):
        atom_nums = []
        try:
            print(f"Processing dataset: {t}")
            data = jdata(t)
            new_data = []
            print(data[0])
            for x in tqdm(data):
                atoms = Atoms(
                    lattice_mat=x['atoms']['lattice_mat'],
                    coords=x['atoms']['coords'],
                    elements=clean_species(x['atoms']['elements']),
                    cartesian=x['atoms']['cartesian'],
                )
                x['structure'] = atoms.pymatgen_converter()
                if x['structure'] is None:
                    continue
                
                # x['structure'] = pymatgen.core.Structure(
                #     lattice=x['atoms']['lattice_mat'],
                #     species=clean_species(x['atoms']['elements']),
                #     coords=x['atoms']['coords'],
                #     coords_are_cartesian=x['atoms']['cartesian'],
                # )

                new_x = {}
                ok = True
                for key in used_vals[i]:
                    newkey, vtype = used_vals[i][key]
                    val = x[key]
                    new_x[newkey] = val
                    if vtype == int and type(val) != int:
                        ok = False
                        break

                    elif vtype == float:
                        if type(val) == int:
                            x[newkey] = float(val)
                        elif type(val) == float and not math.isnan(val) and not math.isinf(val):
                            pass
                        else:
                            ok = False
                            break
                    
                    elif vtype == str and val is None:
                        x[newkey] = ""
                    elif vtype == str and type(val) != str:
                        x[newkey] = str(val)

                if ok:
                    new_data.append(new_x)
                    atom_nums.append(len(x['atoms']['coords']))
            
            
            print(f"filtered: {len(data)} -> {len(new_data)} ({len(new_data) - len(data)})")
            atom_nums = numpy.array(atom_nums).astype(numpy.float64)
            print("Mean Std Median: ", atom_nums.mean(), atom_nums.std(), numpy.median(atom_nums))
            data = new_data
            
            print("Printing the first item...")
            for k in data[0]:
                print(f"{k}\t: {data[0][k]}")

            if t == "megnet":
                id_train, id_val, id_test = get_id_train_val_test(
                    len(data),
                    n_train=60000,
                    n_val=5000,
                    n_test=4239
                )
            else:
                id_train, id_val, id_test = get_id_train_val_test(
                    len(data),
                )

            splits = {}
            splits['train'] = [data[i] for i in id_train]
            splits['val'] = [data[i] for i in id_val]
            splits['test'] = [data[i] for i in id_test]
            splits['all'] = data

            print("Saving split data...")
            for key in splits:
                save_dir = f"{save_names[i]}/{key}/raw"
                os.makedirs(save_dir, exist_ok=True)

                print(f"{key}\t:{len(splits[key])}")
                with open(f"{save_dir}/raw_data.pkl", "wb") as fp:
                    pickle.dump(splits[key], fp)
            
        except Exception as e:
            raise e
