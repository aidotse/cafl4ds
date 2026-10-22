#!/bin/bash
# P1.2.4, card 0, once P1.2.2's confirmation is done: the Places365 off-domain control (3 seeds, matched to P1.2.1's
# IID drive, settings as card 1's COCO control). Card 1 reads both controls on val once this lands.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p124; CARD=0; W=wells/p121/je_heads.pt
run() { ./scripts/run_gaudi_dev.sh -m $TRAIN -m $BDD -m /mnt/cifar100 -m /home/mauricio/data gaudi-env-cafl4ds:latest $CARD "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=je stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
GATE="data=bdd bdd_root=$BDD data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=je pool.split=val pool.fit_fraction=0.4 threads=8 fewshot.lr=3e-5"
TASKS='tasks=[scene,"presence_cell:traffic light"]'

while pgrep -f run_p122_confirm.sh >/dev/null; do sleep 60; done
# A1: the off-domain controls — P1.2.1's IID drive (same well, recipe, 1,748 steps, seeds 0-2), on other images.
CTRL="img_size=224 device=hpu encoder=vit_b16_mae backbone=je data.cache_dir=/workspace/outputs/cache stream.order=iid \
 stream.support_per_canary=10 stream.query_per_canary=10 stream.stratify_canary=false resume_check_split=false"
src=places365
[ -f $OUT/offdomain/drive/$src/checkpoints/s2/final.pt ] || run python scripts/run_deploy_harness.py data=$src $CTRL gate_arms=false \
  eval_every=25 optim.lr=3e-5 well=$W checkpoint_legs=final 'seeds=[0,1,2]' study=P1.2.4 hydra.run.dir=$OUT/offdomain/drive/$src
echo P124_CARD0_DONE
