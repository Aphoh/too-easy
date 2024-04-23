#!/bin/bash

OUT_FILE=./out.csv
export TORCHRUN="torchrun --standalone --nnodes 1 --nproc-per-node 4 --max-restarts 3"
#for MODEL in 410m 
for MODEL in 14m 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b
do
    for REV in main step72000 step36000 step18000 step9000 step4000
    do

        if [[ $MODEL == "14m" || $MODEL == "70m" || $MODEL == "160m" || $MODEL == "410m" || $MODEL == "1b" ]]; then
            BATCH=16
        else
            BATCH=1
        fi

        if ! grep -q "pythia-$MODEL,$REV,stas/c4-en-10k" "$OUT_FILE"; then
            echo "Model $MODEL Rev $REV c4 samples"
            $TORCHRUN -m too_easy.ppl_measurer \
                --dset stas/c4-en-10k --dset-split train \
                --model EleutherAI/pythia-$MODEL --revision $REV \
                --dtype float16 --total-samples 8192 --batch-size $BATCH \
                --output-file $OUT_FILE
        fi        

        if ! grep -q "pythia-$MODEL,$REV,EleutherAI/pythia-memorized-evals" "$OUT_FILE"; then
            echo "Model $MODEL Rev $REV pythia evals"
            $TORCHRUN -m too_easy.ppl_measurer \
                --dset EleutherAI/pythia-memorized-evals --dset-split duped.12b \
                --model EleutherAI/pythia-$MODEL --revision $REV \
                --dtype float16 --total-samples 8192 --batch-size $BATCH \
                --output-file $OUT_FILE
        fi

    done
done