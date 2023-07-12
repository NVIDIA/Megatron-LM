# Use this script to upload data to blob store

# AzureML libraries
from azureml.core import Workspace
from azureml.core.dataset import Dataset
from azureml.data.datapath import DataPath

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

data_dir = "bookcorpus_data"  # Local directory for where data is located that includes .bin and .idx files
blobstore_datadir = data_dir  # Blob store directory to store data in

datastore = ws.get_default_datastore()

# Book Corpus Data
print("upload dataset to blob store")
uploaded_data = Dataset.File.upload_directory(
    src_dir=data_dir,
    target=DataPath(datastore, blobstore_datadir),
    show_progress=True
)

# Usage after uploading the directory
# To refer to the folder directly:
train_dataset = Dataset.File.from_files(path=[(datastore, blobstore_datadir)])
print(train_dataset)
# To refer to a specific file:
# train_dataset = Dataset.File.from_files(path=[(datastore, blobstore_datadir + "/filename.ext")])
# Create DatasetConsumptionConfig to specify how to deliver the dataset to a compute target.
# In the submitted run, files in the datasets will be either mounted or downloaded to local path on the compute target.
# input_data_dir = train_dataset.as_mount()
# input_data_dir = train_dataset.as_download()
