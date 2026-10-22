#!/bin/bash
# P1.2.4, card 1: the JE confirmation (the chosen protections recipe, walk x fresh seeds 1-5) and its val ladder, then
# A1's off-domain controls (COCO x 3, Places365 x 3, matched to P1.2.1's IID drive) and their val reads (frozen rungs).
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p124; CARD=1; W=wells/p121/je_heads.pt
run() { ./scripts/run_gaudi_dev.sh -m $TRAIN -m $BDD -m /mnt/cifar100 -m /home/mauricio/data gaudi-env-cafl4ds:latest $CARD "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=je stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
GATE="data=bdd bdd_root=$BDD data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=je pool.split=val pool.fit_fraction=0.4 threads=8 fewshot.lr=3e-5"
TASKS='tasks=[scene,"presence_cell:traffic light"]'

# The confirmation: P1.2.1's walk recipe plus the protections; fresh seeds (1-5), read as walk_s0..s4 for the gate script.
[ -f $OUT/confirm/drive/je_walk/checkpoints/s5/final.pt ] || run python scripts/run_deploy_harness.py $DIET "$SCENES" gate_arms=false \
  eval_every=25 optim.lr=3e-5 stream.order=walk well=$W checkpoint_legs=final 'seeds=[1,2,3,4,5]' study=P1.2.4 \
  param_groups.layer_decay=0.75 param_groups.no_decay=true hydra.run.dir=$OUT/confirm/drive/je_walk
f=$OUT/confirm/encoders_je.yaml
echo "- {name: b5, kind: well, path: $W}" > $f
for i in 0 1 2 3 4; do echo "- {name: walk_s${i}_end, kind: checkpoint, path: $OUT/confirm/drive/je_walk/checkpoints/s$((i+1))/final.pt}" >> $f; done
[ -f $OUT/confirm/ladder/je/summary.json ] || run python scripts/read_adaptation.py $GATE "$SCENES" "$TASKS" fewshot.enabled=true \
  '~encoders' encoders_file=$f hydra.run.dir=$OUT/confirm/ladder/je

# A1: the off-domain controls — P1.2.1's IID drive (same well, recipe, 1,748 steps, seeds 0-2), on other images.
CTRL="img_size=224 device=hpu encoder=vit_b16_mae backbone=je data.cache_dir=/workspace/outputs/cache stream.order=iid \
 stream.support_per_canary=10 stream.query_per_canary=10 stream.stratify_canary=false resume_check_split=false"
for src in coco places365; do
  [ -f $OUT/offdomain/drive/$src/checkpoints/s2/final.pt ] || run python scripts/run_deploy_harness.py data=$src $CTRL gate_arms=false \
    eval_every=25 optim.lr=3e-5 well=$W checkpoint_legs=final 'seeds=[0,1,2]' study=P1.2.4 hydra.run.dir=$OUT/offdomain/drive/$src
done
f=$OUT/offdomain/encoders.yaml
echo "- {name: b5, kind: well, path: $W}" > $f
for src in coco places365; do for s in 0 1 2; do
  echo "- {name: ${src}_s${s}_end, kind: checkpoint, path: $OUT/offdomain/drive/$src/checkpoints/s$s/final.pt}" >> $f
done; done
[ -f $OUT/offdomain/ladder/summary.json ] || run python scripts/read_adaptation.py $GATE "$SCENES" "$TASKS" fewshot.enabled=false \
  '~encoders' encoders_file=$f hydra.run.dir=$OUT/offdomain/ladder
echo P124_CARD1_DONE
