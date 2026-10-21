#!/bin/bash
# P1.2.1 — the gate read. One worker per card (card 0 JE, card 1 MAE): settle the heads on the frozen off-the-shelf
# encoder, drive one pass of BDD train (walk x 5 seeds, IID x 3), then read the ladder on the untouched BDD val.
# The gate reads scene + cell-balanced traffic-light presence, co-primary (P1.2.0's extended screen).
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p121
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $TRAIN -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
GATE="data=bdd bdd_root=$BDD data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 encoder=vit_b16_mae \
 pool.split=val pool.fit_fraction=0.4 threads=8 fewshot.enabled=true fewshot.lr=3e-5"

encoders() {  # backbone -> the gate's encoder list
  bb=$1; f=$OUT/encoders_$bb.yaml
  echo "- {name: b5, kind: well, path: wells/p121/${bb}_heads.pt}" > $f
  for s in 0 1 2 3 4; do
    d=$OUT/drive/${bb}_walk/checkpoints/s$s
    for c in $(python3 -c "import json;print(' '.join(e['file'][:-3] for e in json.load(open('$d/index.json'))))"); do
      t=${c/step/t}; [ "$c" = final ] && t=end
      ft=false; [ "$t" = end ] && ft=true
      echo "- {name: walk_s${s}_$t, kind: checkpoint, path: $d/$c.pt, fewshot: $ft}" >> $f
    done
  done
  for s in 0 1 2; do
    d=$OUT/drive/${bb}_iid/checkpoints/s$s
    echo "- {name: iid_s${s}_end, kind: checkpoint, path: $d/final.pt}" >> $f
  done
}

worker() {  # card, backbone, lr
  card=$1; bb=$2; lr=$3
  run $card python scripts/warm_well.py $DIET "$SCENES" backbone=$bb optim.lr=3e-4 warmup.freeze_encoder=true \
    warmup.stop=loss warmup.window=200 warmup.tol=0.002 warmup.min_steps=500 warmup.max_steps=3000 \
    warmup.max_passes=100 warmup.select_every=500 well_out=wells/p121/${bb}_heads.pt hydra.run.dir=$OUT/wells/${bb}_heads
  for order in walk iid; do
    seeds='[0,1,2,3,4]'; [ $order = iid ] && seeds='[0,1,2]'
    run $card python scripts/run_deploy_harness.py $DIET "$SCENES" gate_arms=false eval_every=25 backbone=$bb optim.lr=$lr \
      stream.order=$order well=wells/p121/${bb}_heads.pt checkpoint_legs=steps 'checkpoint_steps=[100,0.25,0.5,0.75]' \
      "seeds=$seeds" study=P1.2.1 hydra.run.dir=$OUT/drive/${bb}_$order
  done
  encoders $bb
  run $card python scripts/read_adaptation.py $GATE "$SCENES" backbone=$bb 'tasks=[scene,"presence_cell:traffic light"]' \
    '~encoders' encoders_file=$OUT/encoders_$bb.yaml hydra.run.dir=$OUT/ladder/$bb
}
worker 0 je 3e-5 > $OUT/worker0.log 2>&1 &
worker 1 mae 1e-4 > $OUT/worker1.log 2>&1 &
wait
echo P121_DONE
