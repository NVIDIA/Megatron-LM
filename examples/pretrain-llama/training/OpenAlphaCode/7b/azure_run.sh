# az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
#  --resource-group LLM-Test \
#  --workspace-name Provisioning-Test \
#  --file "examples/pretrain-llama/training/OpenAlphaCode/7b/azureml_conf.yaml"

az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
 --resource-group LLM-Test \
 --workspace-name Provisioning-Test \
 --file "examples/pretrain-llama/training/OpenAlphaCode/7b/azureml_conf_resume.yaml"