#!/bin/bash
# P1.2.4 B1 stacked on the chosen arm (protect): alpha on the selection set, card 1.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; OUT=outputs/p124; CARD=1
run() { ./scripts/run_gaudi_dev.sh -m $TRAIN -m /mnt/cifar100 gaudi-env-cafl4ds:latest $CARD "$@"; }
BASE="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=je stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
f=$OUT/encoders_interp_protect.yaml; W=wells/p121/je_heads.pt; c=$OUT/drive/protect/checkpoints/s0/final.pt
: > $f
for a in 0.25 0.5 0.75; do
  echo "- {name: protect_a$a, kind: interp, alpha: $a, a: {kind: well, path: $W}, b: {kind: checkpoint, path: $c}}" >> $f
done
run python scripts/read_adaptation.py $BASE "$SCENES" pool.kind=selection threads=8 '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/interp_protect
echo P124_INTERP_DONE
