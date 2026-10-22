#!/bin/bash
# P1.2.2's confirmation, card 0: the chosen recipe (512x8 decoder + protections, lr 1e-4) on the walk, fresh seeds 1-5,
# from its own settled-head well; then the val ladder read against P1.2.1's bars (read as walk_s0..s4 for the gate script).
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p122/confirm; CARD=0; W=wells/p122/mae_dec512_heads.pt
run() { ./scripts/run_gaudi_dev.sh -m $TRAIN -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $CARD "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=mae stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
DEC='ssl.decoder_dim=512 ssl.decoder_depth=8 ssl.decoder_heads=16 ssl.decoder_mlp_ratio=4.0'
GATE="data=bdd bdd_root=$BDD data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=mae pool.split=val pool.fit_fraction=0.4 threads=8 fewshot.enabled=true fewshot.lr=3e-5"
[ -f $OUT/drive/mae_walk/checkpoints/s5/final.pt ] || run python scripts/run_deploy_harness.py $DIET "$SCENES" $DEC gate_arms=false \
  eval_every=25 optim.lr=1e-4 stream.order=walk well=$W checkpoint_legs=final 'seeds=[1,2,3,4,5]' study=P1.2.2 \
  param_groups.layer_decay=0.75 param_groups.no_decay=true hydra.run.dir=$OUT/drive/mae_walk
f=$OUT/encoders_mae.yaml
echo "- {name: b5, kind: well, path: $W}" > $f
for i in 0 1 2 3 4; do echo "- {name: walk_s${i}_end, kind: checkpoint, path: $OUT/drive/mae_walk/checkpoints/s$((i+1))/final.pt}" >> $f; done
run python scripts/read_adaptation.py $GATE "$SCENES" $DEC 'tasks=[scene,"presence_cell:traffic light"]' \
  '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder/mae
echo P122_CONFIRM_DONE
