#!/bin/bash
# P1.2.0, third batch: does the few-shot lr picked on B5 disadvantage adapted encoders? The same select_lr grid,
# on each deep well (card 0 JE, card 1 MAE).
set -u
cd /home/mauricio/projects/cafl4ds
TRAIN=/home/mauricio/data/bdd100k; OUT=outputs/p120
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $TRAIN -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card "$@"; }
POOL="data=bdd bdd_root=$TRAIN data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache \
 device=hpu img_size=224 encoder=vit_b16_mae pool.split=train pool.max_images=14000 threads=8"
SCENES='data.scenes=["city street",highway,residential]'
sel() { run $1 python scripts/read_adaptation.py $POOL "$SCENES" backbone=$2 mode=select_lr fewshot.draws=3 '~encoders' \
  "+encoders=[{name:${2}_well,kind:well,path:wells/p110_deep/${2}_o0.0.pt}]" hydra.run.dir=$OUT/lr_select_wells/$2; }
sel 0 je > $OUT/batch3_je.log 2>&1 &
sel 1 mae > $OUT/batch3_mae.log 2>&1 &
wait
echo P120C_DONE
