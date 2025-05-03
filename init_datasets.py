# Utility code to initialize all the datasets.
# It will be particularly useful when running multiple training jobs in parallel.
# In such a case, run this script before starting the jobs.

from dataloaders.dataset_latticeformer import RegressionDatasetMP_Latticeformer as Dataset
from dataloaders.common import CellFormat
splits = ["train", "val", "test"]
datasets = [
    "jarvis__megnet",
    "jarvis__megnet-shear_modulus",
    "jarvis__megnet-bulk_modulus",
    "jarvis__dft_3d_2021",
    "jarvis__dft_3d_2021-ehull",
    "jarvis__dft_3d_2021-mbj_bandgap",
    "jarvis__oqmd_3d",
    "jarvis__oqmd_3d-bandgap",
]
formats = [CellFormat.RAW, CellFormat.PRIMITIVE]
formats = [CellFormat.RAW]

import torch
for dataset in datasets:
    for split in splits:
        for fm in formats:
            print("Processing ------------------", dataset, split, fm)
            data = Dataset(split, dataset, f)
            sizes = data.data.sizes.float()
            print(torch.mean(sizes).item(), torch.max(sizes).item(), torch.median(sizes).item(), torch.std(sizes).item())

