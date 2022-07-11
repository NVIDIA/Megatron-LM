mkdir -p checkpoints/gpt2_345m

cd checkpoints/gpt2_345m
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
unzip megatron_lm_345m_v0.0.zip
rm megatron_lm_345m_v0.0.zip
cd ../..

