for ENG_TOK in {0..10}
do
    AR_TOK=$((10 - $ENG_TOK))
    sed "s/\${{ENG_TOK}}/$ENG_TOK/g; s/allam-7bv3.0_dolma.tune/allam-7bv3.0_dolma.tune.en-$ENG_TOK".ar-"$AR_TOK/g" examples/pretrain-allam/training/allam-7bv3.0-tune/azureml_conf.yaml > examples/pretrain-allam/training/allam-7bv3.0-tune/temp.yaml
    cat examples/pretrain-allam/training/allam-7bv3.0-tune/temp.yaml
    az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
     --resource-group LLM-Test \
     --workspace-name Provisioning-Test \
     --file examples/pretrain-allam/training/allam-7bv3.0-tune/temp.yaml
    rm examples/pretrain-allam/training/allam-7bv3.0-tune/temp.yaml
done