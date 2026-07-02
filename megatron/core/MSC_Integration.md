# Multi-Storage Client (MSC) Integration

The [Multi-Storage Client](https://github.com/NVIDIA/multi-storage-client) (MSC) provides a unified interface for reading datasets and storing checkpoints from both filesystems (such as local disk, NFS, and Lustre) and object storage providers such as S3, Google Cloud Storage (GCS), Oracle Cloud Infrastructure (OCI), Azure, AIStore, and SwiftStack.

This guide covers:

1. Installing and configuring MSC
2. Training models directly from datasets in object storage
3. Saving and loading model checkpoints to and from object storage

## Installation

Install MSC from PyPI as the `multi-storage-client` package.

The base [client](https://nvidia.github.io/multi-storage-client/user_guide/concepts.html#term-client) supports POSIX file systems by default. Install an extra for each storage service to pull in the required dependencies for that provider.

```bash
# POSIX file systems.
pip install multi-storage-client

# AWS S3 and S3-compatible object stores.
pip install "multi-storage-client[boto3]"

# Google Cloud Storage (GCS).
pip install "multi-storage-client[google-cloud-storage]"
```

## Configuration File

MSC uses a YAML configuration file to define how it connects to object storage systems. Each profile represents a different storage backend or bucket, keeping access keys, bucket names, and provider-specific options out of your training scripts.

Here's an example configuration:

```yaml
profiles:
  my-profile:
    storage_provider:
      type: s3
      options:
        # Set the bucket/container name as the base_path
        base_path: my-bucket
        region_name: us-west-2
    # Optional credentials (can also use environment variables for S3)
    credentials_provider:
      type: S3Credentials
      options:
        access_key: ${AWS_ACCESS_KEY}
        secret_key: ${AWS_SECRET_KEY}

cache:
  size: 500G               # Maximum cache size
  location: /tmp/msc_cache # Cache directory on filesystem
```

To tell MSC where to find this file, set the following environment variable before running your Megatron-LM script:

```bash
export MSC_CONFIG=/path/to/msc_config.yaml
```

## MSC URL Format

MSC uses a custom URL scheme to identify and access files across different object storage providers. This scheme makes it easy to reference data and checkpoints without worrying about the underlying storage implementation. An MSC URL has the following structure:

```
msc://<profile-name>/<path/to/object>
```

**Components:**

* `msc://`: The scheme identifier indicating the path should be interpreted by the Multi-Storage Client.
* `<profile-name>`: A named profile defined in your YAML configuration file under the `profiles` section. Each profile specifies the storage provider (such as S3 or GCS), credentials, and storage-specific options such as the bucket name or base path.
* `<path/to/object>`: The logical path to the object or directory within the storage provider, relative to the `base_path` configured in the profile. It behaves similarly to a path in a local filesystem but maps to object keys or blobs in the underlying storage system.

**Example:**

Given the following profile configuration:

```yaml
profiles:
  my-profile:
    storage_provider:
      type: s3
      options:
        base_path: my-bucket
```

The MSC URL:

```
msc://my-profile/dataset/train/data.bin
```

resolves to the object key `dataset/train/data.bin` inside the S3 bucket `my-bucket`. If this were a GCS or OCI profile instead, MSC would apply the appropriate backend logic based on the profile definition, but your code using the MSC URL would remain unchanged.

This abstraction allows training scripts to reference storage resources uniformly—whether they are hosted on AWS, Google Cloud, Oracle, or Azure—just by switching profiles in the config file.


## Train from Object Storage

To train with datasets stored in object storage, use an MSC URL with the `--data-path` argument. This URL references a dataset stored under a profile defined in your MSC configuration file.

In addition, Megatron-LM requires the `--object-storage-cache-path` argument when reading from object storage. Megatron uses this path to cache the `.idx` index files associated with `IndexedDataset` for efficient data access.

```bash
python pretrain_gpt.py                                      \
    --object-storage-cache-path /path/to/object_store_cache \
    --data-cache-path /path/to/data_cache                   \
    --data-path msc://my-profile/datasets/text_document     \
    --no-mmap-bin-files
```

**Note:** All four arguments must be provided when training with datasets in object storage using MSC.

## Save and Load Checkpoints from Object Storage

MSC can be used to save and load model checkpoints directly from object storage by specifying MSC URLs for the `--save` and `--load` arguments. This allows you to manage checkpoints in object storage.

```bash
python pretrain_gpt.py                \
  --save msc://my-profile/checkpoints \
  --load msc://my-profile/checkpoints \
  --save-interval 1000
```

**Note:** MSC URLs currently support only the `torch_dist` checkpoint format for saving and loading.

## Enable MSC

MSC integration is opt-in: even when the `multi-storage-client` library is installed, MSC is **disabled by default**. To opt in, pass the `--enable-msc` flag. Once enabled, MSC is also used for regular filesystem paths (like `/filesystem_mountpoint/path` in `--data-path`, `--save`, or `--load`), not just explicit `msc://` URLs.

```bash
python pretrain_gpt.py --enable-msc
```

> **Note:** When MSC is enabled, the dist-checkpointing loader uses `msc.torch.MultiStorageFileSystemReader` instead of `CachedMetadataFileSystemReader`. This means MSC silently overrides `ckpt_assume_constant_structure=True` (and any other path that requests `cache_metadata=True`) and re-reads metadata on every load. MSC emits a warning when this occurs.

## Performance Considerations

Object storage introduces performance trade-offs compared to local filesystems. When using object storage with MSC, there are a few important performance implications to keep in mind:

**Reading Datasets**

Reading training datasets directly from object storage is typically slower than reading from local disk. This is primarily due to:
* High latency of object storage systems, especially for small and random read operations (for example, reading samples from `.bin` files).
* HTTP-based protocols used by object stores (for example, S3 GET with range requests), which are slower than local filesystem I/O.

To compensate for this latency, increase the number of data loading workers using the `--num-workers` argument:

```
python pretrain_gpt.py --num-workers 8 ...
```

Increasing the number of workers allows more parallel reads from object storage, helping to mask I/O latency and maintain high GPU utilization during training.

**Checkpoint Loading**

When loading checkpoints from object storage, configure the `cache` section in your MSC configuration file. The local cache stores downloaded checkpoint data and metadata, which significantly reduces load time and memory usage.

**Example:**

```
cache:
  size: 500G
  location: /tmp/msc_cache
```

For optimal performance, configure the cache directory on a high-speed local storage device such as an NVMe SSD.

## Additional Resources and Advanced Configuration

Refer to the [MSC Configuration Documentation](https://nvidia.github.io/multi-storage-client/references/configuration.html) for details on supported storage providers, credentials, and caching strategies.

MSC supports collecting observability metrics and traces to help monitor and debug data access patterns during training. These metrics can help you identify bottlenecks in your data loading pipeline, optimize caching strategies, and monitor resource utilization when training with large datasets in object storage. For more information about MSC's observability features, refer to the [MSC Observability Documentation](https://nvidia.github.io/multi-storage-client/user_guide/telemetry.html).

MSC offers an experimental Rust client that bypasses Python's Global Interpreter Lock (GIL) to significantly improve performance for multi-threaded I/O operations. The Rust client supports AWS S3, SwiftStack, and Google Cloud Storage, enabling true concurrent execution for much better performance compared to the Python implementation. To enable it, add `rust_client: {}` to your storage provider configuration. For more details, refer to the [MSC Rust Client Documentation](https://nvidia.github.io/multi-storage-client/user_guide/rust.html).
