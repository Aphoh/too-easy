#!/bin/bash

MODEL=70m
OUT_FILE=./70m-c4-f16.csv
for REV in main step72000 step36000 step18000 step9000 step4000 step2000
do

    if ! grep -q "pythia-$MODEL,$REV,stas/c4-en-10k" "$OUT_FILE"; then
        echo "Model $MODEL Rev $REV c4 samples"
        python -m too_easy.ppl_measurer \
            --dset stas/c4-en-10k --dset-split train \
            --model EleutherAI/pythia-$MODEL --revision $REV \
            --dtype float16 --total-samples 32 --batch-size 1 \
            --output-file $OUT_FILE --device cpu
    fi        
done