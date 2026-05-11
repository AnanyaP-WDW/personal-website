---
title: Reducing OOM in ML Pipelines with Parquet + PyArrow + Streaming Standardization
date: 2026-05-05
description: A practical pattern for preventing OOMKilled training jobs using Parquet storage, PyArrow batch scanning, and batch-wise standardization.
tags: ["MLOps", "Parquet", "PyArrow", "Kubernetes", "Machine Learning"]
---

# Reducing OOM in ML Pipelines with Parquet + PyArrow + Streaming Standardization

## Index

1. [Why OOM Happens in Training Jobs](#why-oom-happens-in-training-jobs)
2. [Why Parquet Helps](#why-parquet-helps)
3. [How PyArrow Enables Streaming](#how-pyarrow-enables-streaming)
4. [Memory Logic: Eager vs Streaming](#memory-logic-eager-vs-streaming)
5. [StandardScaler in Batches (Math)](#standardscaler-in-batches-math)
6. [Two-Pass Preprocessing Pattern](#two-pass-preprocessing-pattern)
7. [How This Fits Model Training](#how-this-fits-model-training)
8. [Practical Benefits](#practical-benefits)
9. [Simple Checklist](#simple-checklist)

---

## Why OOM Happens in Training Jobs

Troubleshooting out of memory (OOM) errors in MLops systems are often not talked about and overshadowed by more trendy applications like model training, architecture, tuning etc. Running real world data engineering and ML pipelines on a k8 cluster can be a tedious process, resulting in hours of troubleshooting errors and bugs which require tediously reading traces. One such issue that I have constantly faced over the years in the OOMKilled error. It's a k8's status that forcefully terminated the linux kernel because a process exceeded the allocated memory on the host node. One solution for this problem is to incerease the hw limits of your pod or to split the job across pods. A more rigourous solution is to reduce both interstitial and final object memory - this is where paraquet and arrow are used. 
In practice OOMKilled usually happens when a ML pipeline loads all splits (`train`, `val`, `test`) into memory at once, then creates additional copies or during complex ETL or data trasnformation jobs. These objects can be of differnt types:

- S3 bytes in memory
- NumPy arrays
- PyTorch tensors
- Temporary transformed copies

If data is large, peak memory can be several times larger than the dataset itself.

---

## Why Parquet Helps

Parquet is a columnar, compressed storage format designed to cut peak object memory.

Key benefits:

- Stores columns efficiently (good compression)
- Reads only needed columns when possible
- Supports chunked reading
- Good fit for analytical/ML tabular data

Example: unlike loading a massive `.npy` file all at once, Parquet lets you process data incrementally in manageable chunks.

---

## How PyArrow Enables Streaming

`pyarrow` gives fast Parquet readers/writers and record-batch iteration.

Typical pattern that I use for building such systems:

1. List Parquet part files under an S3/MinIO prefix
2. Open one part file
3. Iterate in record batches (`batch_size`, e.g. 50k rows)
4. Process and release each batch

So memory is roughly bounded by:

$$
\text{Peak RAM} \approx \text{one batch} + \text{model state} + \text{small overhead}
$$

not by full dataset size.

---

## Memory Logic: Eager vs Streaming

### Eager loading

$$
X_{train}, y_{train}, X_{val}, y_{val}, X_{test}, y_{test} \;\text{all loaded together}
$$

Then converted to tensors:

$$
\text{RAM} \uparrow \text{ again due to tensor copies}
$$

### Streaming loading

Only one batch is active:

$$
(X_b, y_b),\; b = 1,2,\dots,B
$$

After each batch:

- forward/backward/update
- batch released
- next batch loaded

This keeps peak memory stable and much lower.

Both paraquet and pyarrow are highly compatible with pandas and polars. 
The flow I generally use is:
- Parquet bytes are read with ```pyarrow.parquet.ParquetFile(...)```
- Iteration happens as Arrow RecordBatch objects via ```iter_batches(...)```
- Then each batch is converted to pandas with ```.to_pandas()``` method
So it is: Parquet -> PyArrow batch -> pandas DataFrame.
This is how I scan Parquet datasets from S3 and yielding Arrow RecordBatches which in turn can be converted to pandas using:

```py
def scan_parquet_dataset(
    bucket: str,
    prefix: str,
    columns: Optional[List[str]] = None,
    batch_size: int = DEFAULT_SCAN_BATCH_SIZE,
    s3_client=None,
) -> Iterator[pa.RecordBatch]:
    """Lazily scan a Parquet dataset on S3, yielding Arrow RecordBatches.

    Iterates over part files in sorted order.  Within each file the table
    is sliced into batches of *batch_size* rows.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    logger.info("parquet_scan:list:start bucket=%s prefix=%s", bucket, prefix)
    keys = _list_parquet_keys(bucket, prefix, s3_client)
    logger.info(
        "parquet_scan:list:done bucket=%s prefix=%s parquet_files=%d",
        bucket,
        prefix,
        len(keys),
    )

    for idx, key in enumerate(keys, start=1):
        logger.info("parquet_scan:file:start index=%d total=%d key=%s", idx, len(keys), key)
        yielded = 0
        for rb in _iter_parquet_record_batches_from_s3(
            bucket,
            key,
            s3_client,
            columns=columns,
            batch_size=batch_size,
        ):
            yielded += 1
            yield rb
        logger.info(
            "parquet_scan:file:done index=%d total=%d key=%s batches=%d",
            idx,
            len(keys),
            key,
            yielded,
        )


def scan_parquet_as_pandas(
    bucket: str,
    prefix: str,
    columns: Optional[List[str]] = None,
    batch_size: int = DEFAULT_SCAN_BATCH_SIZE,
    s3_client=None,
) -> Iterator[pd.DataFrame]:
    """Lazily scan a Parquet dataset on S3, yielding pandas DataFrames."""
    for idx, rb in enumerate(
        scan_parquet_dataset(
        bucket, prefix, columns=columns,
        batch_size=batch_size, s3_client=s3_client,
        ),
        start=1,
    ):
        logger.info("parquet_scan:to_pandas:start batch=%d rows=%d", idx, rb.num_rows)
        df = rb.to_pandas()
        logger.info(
            "parquet_scan:to_pandas:done batch=%d rows=%d cols=%d",
            idx,
            len(df),
            len(df.columns),
        )
        yield df
```

---

## StandardScaler in Batches (Math)

For feature standardization, we want:

$$
z = \frac{x - \mu}{\sigma}
$$

where:

- $$\mu$$: mean of training data feature
- $$\sigma$$: standard deviation of training data feature

### Problem

Computing $$\mu$$ and $$\sigma$$ from all rows at once may be memory-heavy.

### Batch-wise solution (`partial_fit`)

Process mini-batches and update running statistics.

For one feature and one batch $$b$$:

- batch size: $$n_b$$
- batch mean: $$\mu_b$$
- batch variance: $$\sigma_b^2$$

Running totals:

$$
N = \sum_b n_b
$$

Global mean (conceptually):

$$
\mu = \frac{1}{N}\sum_b n_b\mu_b
$$

Variance can also be merged from batch statistics (what incremental scaler implementations do internally).

So we get the same global normalization behavior without loading all rows at once.

---

## Two-Pass Preprocessing Pattern

A simple and robust pattern:

### Pass 1: Fit scaler only

- Scan training Parquet in batches
- Extract numeric columns
- Call `partial_fit(batch)`

### Pass 2: Transform and save

- Scan again in batches
- Apply scaler transform
- Write transformed batches to Parquet
- Optionally export `.npy` for backward compatibility

This separates "learn normalization parameters" from "apply normalization", while keeping memory bounded.

---

## How This Fits Model Training

With Parquet metadata (prefixes) instead of giant loaded arrays:

- `train` DataLoader streams from `parquet/train`
- `val` streams from `parquet/val`
- `test` streams from `parquet/test`

Model code can stay mostly the same because it still receives `(features, labels)` batches.

---

## Practical Benefits

- Lower peak memory, fewer OOMKilled pods
- Better scalability to larger datasets
- Cleaner separation of storage and compute
- Works well with S3/MinIO object storage
- Easier to tune batch sizes for RAM constraints

---

## Simple Checklist

- Store preprocessed splits as Parquet (`train`, `val`, `test`)
- Keep prefixes in manifest metadata
- Use PyArrow batch scanning (`scan_parquet_as_pandas` / record batches)
- Fit scaler with `partial_fit` over train batches
- Stream DataLoader from Parquet instead of loading full arrays
- Keep legacy `.npy` fallback only if needed

---

Parquet + PyArrow + batch-wise standardization is a practical way to make ML pipelines memory-safe in Kubernetes, especially when training data grows over time.
