#!/bin/bash

set -e -x

input_type=PyTorch

for i in {0..1}
do
    kineto_file=`ls result/$i`
    # echo $kineto_file
    chakra_trace_link --chakra-host-trace pytorch_et_${i}.json --chakra-device-trace $(dirname result)/result/$i/$kineto_file --output-file pytorch_et_${i}_plus.json
    chakra_converter $input_type --input pytorch_et_${i}_plus.json --output final.${i}.et
    chakra_jsonizer --input_filename final.${i}.et --output_filename final.${i}.json
done
chakra_pg_extractor --input_filename ./final --output_filename comm.json
