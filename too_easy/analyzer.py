import click
import glob

import dask.array as da
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


@click.command()
@click.argument("zarrs", type=click.Path(exists=True))
def analyze_zarr(zarrs):
    # Read the .zarr file into a Dask array
    zarr_arrs = []
    steps = []
    for file in glob.glob(zarrs + "/step*.zarr"):
        zarr_arrs.append(da.from_zarr(file))
        name = Path(file).stem[4:]
        steps.append(int(name))
    bins = torch.load(Path(zarrs).parent / "bins.pt").numpy()

    output_arr = da.stack(zarr_arrs, axis=0)

    # Call your function on the Dask array
    make_plots(output_arr, steps, bins)


def make_plots(dask_array, steps, bins):
    # Implement your function logic here
    # This is just a placeholder example
    b0_idx = np.where(bins == 0)[0][0]
    gt0s = da.sum(dask_array[:, :, b0_idx:], axis=(1,2)) / da.sum(dask_array, axis=(1,2))
    gt0s = gt0s.compute()
    plt.plot(steps, gt0s, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Fraction of values > 0")
    plt.savefig("output-gt0-steps.png")

if __name__ == "__main__":
    analyze_zarr()
