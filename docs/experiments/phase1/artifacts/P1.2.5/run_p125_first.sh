#!/bin/bash
# P1.2.5, the first runs — everything that neither reads an adapted encoder nor depends on DINO's objective settings.
# Card 0: the few-shot learning rate per frozen floor (B5-DINO, the supervised ViT-B; B5 keeps P1.2.0's 3e-5), Part C's
#   screen on frozen DINO, then Part D's off-domain controls (COCO x 3, Places365 x 5).
# Card 1: the settled-head wells on the enlarged selection set (MAE-init JE; DINO-init SimSiam), Part B's MAE-init walk
#   re-run x 5 with curve checkpoints, then Part D's BDD IID drive x 5.
# Every drive uses P1.2.4's confirmed protections recipe (lr 3e-5, layer decay 0.75, no decay on norms / biases /
# positions / tokens). The controls are matched by step count to the shorter drive: 1,692 x 32 + 40 probes = 54,184.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; GEO=$TRAIN/geometry; OUT=outputs/p125
mkdir -p $OUT wells/p125
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/pr_2018_scaleout_workdir/cafl4ds/data/cifar100 -m /mnt/pr_2018_scaleout_workdir/cafl4ds/data \
  -m /home/mauricio/.cache/torch gaudi-env-cafl4ds:latest $card "$@"; }
# P1.2.0's train-side pool (the first 14,000 frames of BDD train), with the geometry labels (train's file has box2d)
POOL="data=bdd bdd_root=$TRAIN data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache \
 device=hpu img_size=224 pool.split=train pool.max_images=14000 threads=8 readout=mean \
 data.boxes=true data.drivable_dir=$GEO/drivable/labels/train"
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN \
 device=hpu img_size=224 stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=450 stream.select_query_per_canary=450"
PROTECT="gate_arms=false eval_every=25 optim.lr=3e-5 param_groups.layer_decay=0.75 param_groups.no_decay=true study=P1.2.5"
CANDS='screen.candidates=["presence_cell:bus","presence_cell:truck","presence_cell:pedestrian","large_cell:car,bus,truck:0.02",weather,drivable]'
W=wells/p125/mae_je_heads.pt

card0() {
  for fl in dino:vit_b16_dino sup:vit_b16_pretrained; do
    name=${fl%%:*}; enc=${fl#*:}
    [ -f $OUT/lr/$name/lr_select.json ] || run 0 python scripts/read_adaptation.py $POOL "$SCENES" encoder=$enc mode=select_lr \
      fewshot.select_task=scene fewshot.select_k=100 'fewshot.lr_grid=[1e-5,3e-5,1e-4]' hydra.run.dir=$OUT/lr/$name
  done
  lr=$(python3 -c "import json;print(json.load(open('$OUT/lr/dino/lr_select.json'))['best_lr'])")
  [ -f $OUT/screen/screen.json ] || run 0 python scripts/read_adaptation.py $POOL "$SCENES" encoder=vit_b16_dino mode=screen \
    fewshot.lr=$lr fewshot.select_k=100 fewshot.draws=3 screen.reference=scene "$CANDS" screen.max_overlap=0.60 \
    screen.min_headroom=0.05 screen.max_tasks=4 hydra.run.dir=$OUT/screen
  # Part D's off-domain controls, from the MAE-init well (card 1 builds it first)
  while [ ! -f $W ]; do sleep 60; done
  CTRL="img_size=224 device=hpu encoder=vit_b16_mae backbone=je data.cache_dir=/workspace/outputs/cache stream.order=iid \
   stream.support_per_canary=10 stream.query_per_canary=10 stream.stratify_canary=false resume_check_split=false \
   data.subset=54184 data.subset_seed=0"
  for sc in coco:'[0,1,2]':s2 places365:'[0,1,2,3,4]':s4; do
    IFS=: read src seeds last <<< "$sc"
    [ -f $OUT/offdomain/drive/$src/checkpoints/$last/final.pt ] || run 0 python scripts/run_deploy_harness.py data=$src $CTRL \
      $PROTECT well=$W checkpoint_legs=final "seeds=$seeds" hydra.run.dir=$OUT/offdomain/drive/$src
  done
}

card1() {
  for w in mae_je:vit_b16_mae dino_simsiam:vit_b16_dino; do
    name=${w%%:*}; enc=${w#*:}
    [ -f wells/p125/${name}_heads.pt ] || run 1 python scripts/warm_well.py $DIET "$SCENES" encoder=$enc backbone=je optim.lr=3e-4 \
      warmup.freeze_encoder=true warmup.stop=loss warmup.window=200 warmup.tol=0.002 warmup.min_steps=500 \
      warmup.max_steps=3000 warmup.max_passes=100 warmup.select_every=500 well_out=wells/p125/${name}_heads.pt \
      hydra.run.dir=$OUT/wells/$name
  done
  # Part B: the MAE-init confirmed recipe re-run with the curve's checkpoints (fresh seeds 1-5, as P1.2.4's confirmation)
  [ -f $OUT/drive/mae_je_walk/checkpoints/s5/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET "$SCENES" \
    encoder=vit_b16_mae backbone=je $PROTECT stream.order=walk well=$W checkpoint_legs=steps \
    'checkpoint_steps=[100,0.125,0.25,0.5,0.75]' 'seeds=[1,2,3,4,5]' hydra.run.dir=$OUT/drive/mae_je_walk
  # Part D: the BDD IID drive the controls are contrasted with
  [ -f $OUT/drive/mae_je_iid/checkpoints/s5/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET "$SCENES" \
    encoder=vit_b16_mae backbone=je $PROTECT stream.order=iid well=$W checkpoint_legs=final 'seeds=[1,2,3,4,5]' \
    hydra.run.dir=$OUT/drive/mae_je_iid
}

card0 > $OUT/first_card0.log 2>&1 &
card1 > $OUT/first_card1.log 2>&1 &
wait
echo P125_FIRST_DONE
