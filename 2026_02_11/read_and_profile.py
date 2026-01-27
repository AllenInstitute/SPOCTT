import anndata
import multiprocessing
import os
import psutil
import time


def read(path, axis, idx):
    """
    Read and subset and h5ad file in a multiprocessing.Process
    so that you can track the memory used, independent of the
    parent process.

    Print the time it took and the maximum memory usage to stdout.

    Parameters
    ----------
    path:
        the path to the h5ad file
    axis:
        either 'cell' or 'gene'; the axis along which
        to subset the data in the h5ad file
    idx:
        np.array of integers; the rows/columns to grab
        in the subset
    """
    print(f"slicing {path.name} along '{axis}'")
    max_mem = 0.0
    p = multiprocessing.Process(
        target=_read,
        kwargs={'axis': axis, 'path': path, 'idx': idx}
    )
    p.start()
    while p.exitcode is None:
        try:
            mem = psutil.Process(p.pid).memory_info().rss
            if mem > max_mem:
                max_mem = mem
        except Exception:
            break
    gb = max_mem/(1024**3)
    p.join()
    file_size = os.stat(path).st_size / (1024**3)
    print(
        f"maximum memory footprint {gb:.2e} GB "
        f"(file size is {file_size:.2e} GB)")

def _read(path, axis, idx):
    src = anndata.read_h5ad(path, backed='r')

    if axis == 'gene':
        t0 = time.time()
        subset = src[:, idx].to_memory()
        dur = time.time()-t0
    elif axis == 'cell':
        t0 = time.time()
        subset = src[idx, :].to_memory()
        dur = time.time()-t0
    dur *= 1000.0
    print(f'took {dur:.2e} ms')
    
    src.file.close()
    del src
    del subset


def read_by_chunks(path, gene_idx, rows_at_a_time=5000):
    """
    Read and subset an h5ad file along the gene axis, reading
    it in a chunk of rows at a time and concatenating the
    resulting sub-AnnData objects into one AnnData object.

    Use a multiprocessing.Process so you can track the memory
    footprint independent of the memory footprint of the
    parent process.

    Print the amount of time it took and the memory footprint
    to stdout.

    Parameters
    ----------
    path:
        path to the h5ad file
    gene_idx:
        np.array of ints; the genes to grab in the subset
    rows_at_a_time:
        the number of cells to read in at a time
    """
    print(f"slicing {path.name} along axis 'gene'")
    max_mem = 0.0
    p = multiprocessing.Process(
        target=_read_by_chunks,
        kwargs={
            'path': path,
            'gene_idx': gene_idx,
            'rows_at_a_time': rows_at_a_time
        }
    )
    p.start()
    while p.exitcode is None:
        try:
            mem = psutil.Process(p.pid).memory_info().rss
            if mem > max_mem:
                max_mem = mem
        except Exception:
            break
    gb = max_mem/(1024**3)
    p.join()
    file_size = os.stat(path).st_size / (1024**3)
    print(
        f"maximum memory footprint {gb:.2e} GB "
        f"(file size is {file_size:.2e} GB)")


def _read_by_chunks(path, gene_idx, rows_at_a_time=5000):
    t0 = time.time()
    src = anndata.read_h5ad(path, backed='r')
    var = src.var.iloc[gene_idx]

    row_iterator = src.chunked_X(rows_at_a_time)

    sub_adata = []
    for chunk in row_iterator:
        obs_chunk = src.obs[chunk[1]: chunk[2]]
        x_chunk = chunk[0][:, gene_idx]
        sub_adata.append(
            anndata.AnnData(
                obs=obs_chunk,
                var=var,
                X=x_chunk
            )
        )

    result = anndata.concat(sub_adata, merge="same")
    dur = (time.time()-t0)
    print(f"took {dur:.2e} s")
