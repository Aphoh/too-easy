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
    xv = []
    for file in glob.glob(zarrs + "/*.zarr"):
        zarr_arrs.append(da.from_zarr(file))
        name = Path(file).stem
        assert name[-1] == "m", "Expected 'm' as last character in name" 
        xv.append(int(name[:-1]))
    bins = torch.load(Path(zarrs) / "bins.pt").numpy()

    by_xv = sorted(zip(xv, zarr_arrs), key=lambda x: x[0])
    xv, zarr_arrs = list(zip(*by_xv))

    # Call your function on the Dask array
    make_plots(zarr_arrs, xv, bins)


def make_plots(dask_array, xv, bins):
    # Implement your function logic here
    # This is just a placeholder example
    b0_idx = np.where(bins == 0)[0][0]
    gt0s = []
    for arr in dask_array:
        gt0s.append((da.sum(arr[:, b0_idx:]) / da.sum(arr)).compute())
    #gt0s = da.sum(dask_array[:, :, b0_idx:], axis=(1, 2)) / da.sum(dask_array, axis=(1, 2))
    #gt0s = gt0s.compute()
    plt.plot(xv, gt0s, marker="o")
    plt.xlabel("Model size")
    plt.xscale("log")
    plt.ylabel("Fraction of values > 0")
    plt.show()
    plt.savefig("output-gt0-steps.png")


if __name__ == "__main__":
    analyze_zarr()
