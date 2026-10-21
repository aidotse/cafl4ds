#!/bin/bash
# P1.2.0 — card 0: few-shot lr on B5, presence screen, JE ladder; card 1: head-settle scouts (MAE, JE), MAE ladder,
# then the canary-pool reads for both backbones.
set -u
cd /home/mauricio/projects/cafl4ds
TRAIN=/home/mauricio/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p120
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $TRAIN -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card "$@"; }
POOL="data=bdd bdd_root=$TRAIN data.min_canary_count=50 device=hpu img_size=224 encoder=vit_b16_mae \
 pool.split=train pool.max_images=14000 threads=8"
TASKS='tasks=[scene,"presence:traffic light","presence:pedestrian","presence:truck"]'
DIET="data=bdd data.split=val data.min_canary_count=50 bdd_root=$BDD device=hpu img_size=224 encoder=vit_b16_mae \
 stream.support_per_canary=45 stream.query_per_canary=45 stream.stratify_canary=true stream.well_fraction=0.5"
SCOUT="stream.select_support_per_canary=50 stream.select_query_per_canary=50 warmup.drive_read_per_class=500"

encoders() {  # backbone -> encoders yaml (B5 + random only in the JE list)
  bb=$1; f=$OUT/encoders_$bb.yaml; : > $f
  [ $bb = je ] && printf -- "- {name: b5, kind: pretrained}\n- {name: random, kind: random}\n" >> $f
  echo "- {name: ${bb}_well, kind: well, path: wells/p110_deep/${bb}_o0.0.pt}" >> $f
  for order in walk iid; do for s in 0 1 2 3 4; do
    d=outputs/p110_deep/warm/${bb}_o0.0_$order/checkpoints/s$s
    echo "- {name: ${bb}_${order}_s${s}_mid, kind: checkpoint, path: $d/step00060.pt, fewshot: false}" >> $f
    echo "- {name: ${bb}_${order}_s${s}_end, kind: checkpoint, path: $d/final.pt}" >> $f
  done; done
}
lr() { python3 -c "import json;print(json.load(open('$OUT/lr_select/lr_select.json'))['best_lr'])"; }
wait_lr() { while [ ! -f $OUT/lr_select/lr_select.json ]; do sleep 60; done; }

scout() {  # card, backbone, drive lr
  card=$1; bb=$2; dlr=$3
  run $card python scripts/warm_well.py $DIET $SCOUT backbone=$bb optim.lr=3e-4 warmup.freeze_encoder=true \
    warmup.stop=loss warmup.window=200 warmup.tol=0.002 warmup.min_steps=500 warmup.max_steps=3000 \
    warmup.max_passes=100 warmup.select_every=100 well_out=wells/p120/${bb}_heads.pt hydra.run.dir=$OUT/scout/${bb}_heads
  run $card python scripts/warm_well.py $DIET $SCOUT backbone=$bb optim.lr=$dlr resume_well=wells/p120/${bb}_heads.pt \
    warmup.stop=budget warmup.max_steps=1000 warmup.max_passes=100 warmup.select_every=25 \
    well_out=wells/p120/${bb}_unfrozen.pt hydra.run.dir=$OUT/scout/${bb}_unfrozen
}
ladder() {  # card, backbone
  card=$1; bb=$2; encoders $bb
  run $card python scripts/read_adaptation.py $POOL "$TASKS" backbone=$bb fewshot.enabled=true 'fewshot.tasks=[scene]' \
    fewshot.lr=$(lr) '~encoders' encoders_file=$OUT/encoders_$bb.yaml hydra.run.dir=$OUT/ladder/$bb
}
canary() {  # card, backbone — continuity read on P1.1.0's canary probes (val)
  card=$1; bb=$2
  run $card python scripts/read_adaptation.py $DIET stream.select_support_per_canary=150 stream.select_query_per_canary=150 pool.kind=canary backbone=$bb fewshot.enabled=true \
    fewshot.lr=$(lr) probe_per_class=1000 threads=8 '~encoders' encoders_file=$OUT/encoders_$bb.yaml \
    hydra.run.dir=$OUT/canary/$bb
}

# Re-run 2026-10-21: the lr pick, screen and scouts completed; the ladder / canary reads crashed on the
# `~encoders` lookup (fixed in 01ee493). Only the reads are re-run here.
reads() { card=$1; bb=$2; ladder $card $bb; canary $card $bb; }
reads 0 je > $OUT/reads_je.log 2>&1 &
reads 1 mae > $OUT/reads_mae.log 2>&1 &
wait
echo P120_READS_DONE
