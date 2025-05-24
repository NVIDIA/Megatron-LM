## Multi-Storage Client (MSC) Integration

The [Multi-Storage Client](https://github.com/NVIDIA/multi-storage-client) (MSC) provides a unified interface for reading datasets and storing checkpoints from both filesystems (e.g., local disk, NFS, Lustre) and object storage providers such as S3, GCS, OCI, Azure, AIStore, and SwiftStack.

This guide will walk you through how to:

1. How to install and configure MSC
2. How to train models directly using datasets in object storage
3. How to save and load model checkpoints to/from object storage

### Installation

To install the Multi-Storage Client package:

```bash
pip install multi-storage-client
```

For S3 access, you'll also need to install boto3:

```bash
pip install multi-storage-client[boto3]
```

### Configuration File

MSC uses a YAML configuration file to define how it connects to object storage systems. This design allows you to specify one or more storage profiles, each representing a different storage backend or bucket. MSC keeps your training scripts clean and portable by centralizing details in a config file. There is no need to hardcode access keys, bucket names, or other provider-specific options directly into your code.

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
  # Maximum cache size
  size: 500G
  cache_backend:
    # Cache directory on filesystem
    cache_path: /tmp/msc_cache
```

To tell MSC where to find this file, set the following environment variable before running your Megatron-LM script:

```bash
export MSC_CONFIG=/path/to/msc_config.yaml
```

### MSC URL Format

MSC uses a custom URL scheme to identify and access files across different object storage providers. This scheme makes it easy to reference data and checkpoints without worrying about the underlying storage implementation. An MSC URL has the following structure:

```
msc://<profile-name>/<path/to/object>
```

**Components:**

* `msc://` This is the scheme identifier indicating the path should be interpreted by the Multi-Storage Client.
* `<profile-name>` This corresponds to a named profile defined in your YAML configuration file under the profiles section. Each profile specifies the storage provider (e.g., S3, GCS), credentials, and storage-specific options such as the bucket name or base path.
* `<path/to/object>` This is the logical path to the object or directory within the storage provider, relative to the base_path configured in the profile. It behaves similarly to a path in a local filesystem but maps to object keys or blobs in the underlying storage system.

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

is interpreted as accessing the object with the key `dataset/train/data.bin` inside the S3 bucket named `my-bucket`. If this were a GCS or OCI profile instead, MSC would apply the appropriate backend logic based on the profile definition, but your code using the MSC URL would remain unchanged.

This abstraction allows training scripts to reference storage resources uniformly—whether they're hosted on AWS, GCP, Oracle, or Azure—just by switching profiles in the config file.


### Train from Object Storage

To train with datasets stored in object storage, use an MSC URL with the `--data-path` argument. This URL references a dataset stored under a profile defined in your MSC configuration file.

In addition, Megatron-LM requires the `--object-storage-cache-path` argument when reading from object storage. This path is used to cache the `.idx` index files associated with IndexedDataset, which are needed for efficient data access.

```bash
python pretrain_gpt.py                                      \
    --object-storage-cache-path /path/to/object_store_cache \
    --data-cache-path /path/to/data_cache                   \
    --data-path msc://my-profile/datasets/text_document     \
    --no-mmap-bin-files
```

**NOTE:** All four arguments must be provided when training with datasets in object storage using MSC.

### Save and Load Checkpoints from Object Storage

MSC can be used to save and load model checkpoints directly from object storage by specifying MSC URLs for the `--save` and `--load` arguments. This allows you to manage checkpoints in object storage.

```bash
python pretrain_gpt.py                \
  --save msc://my-profile/checkpoints \
  --load msc://my-profile/checkpoints \
  --save-interval 1000
```

**Notes:** Only the `torch_dist` checkpoint format is currently supported when saving to or loading from MSC URLs.

### Disable MSC

By default, MSC integration is automatically enabled when the `multi-storage-client` library is installed. MSC is also used for regular filesystem paths (like `/filesystem_mountpoint/path` in `--data-path`, `--save`, or `--load`) even when not using explicit MSC URLs. MSC functions as a very thin abstraction layer with negligible performance impact when used with regular paths, so there's typically no need to disable it. If you need to disable MSC, you can do so using the `--disable-msc` flag:

```bash
python pretrain_gpt.py --disable-msc
```

### Performance Considerations

When using object storage with MSC, there are a few important performance implications to keep in mind:

**Reading Datasets**

Reading training datasets directly from object storage is typically slower than reading from local disk. This is primarily due to:
* High latency of object storage systems, especially for small and random read operations (e.g., reading samples from .bin files).
* HTTP-based protocols used by object stores (e.g., S3 GET with range requests), which are slower than local filesystem I/O.

To compensate for this latency, it is recommended to increase the number of data loading workers using the `--num-workers` argument in your training command:

```
python pretrain_gpt.py --num-workers 8 ...
```

Increasing the number of workers allows more parallel reads from object storage, helping to mask I/O latency and maintain high GPU utilization during training.

**Checkpoint Loading**

When using MSC to load checkpoints from object storage, it is important to configure the cache section in your MSC configuration file. This local cache is used to store downloaded checkpoint data and metadata, which significantly reduces load time and memory usage.

Example:

```
cache:
  size: 500G
  cache_backend:
    cache_path: /tmp/msc_cache
```

Make sure this cache directory is located on a fast local disk (e.g., NVMe SSD) for optimal performance.

### Additional Resources and Advanced Configuration

Refer to the [MSC Configuration Documentation](https://nvidia.github.io/multi-storage-client/config/index.html) for complete documentation on MSC configuration options, including detailed information about supported storage providers, credentials management, and advanced caching strategies.

MSC also supports collecting observability metrics and traces to help monitor and debug data access patterns during training. These metrics can help you identify bottlenecks in your data loading pipeline, optimize caching strategies, and monitor resource utilization when training with large datasets in object storage.

For more information about MSC's observability features, see the [MSC Observability Documentation](https://nvidia.github.io/multi-storage-client/config/index.html#opentelemetry).
