#!/bin/bash
# P1.2.2 — MAE recipe diagnosis, card 0. The zero-lr pipeline check, then the one-seed recipe arms (walk, seed 0,
# from P1.2.1's settled-head MAE well), each read on the train-side selection set (never val).
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; OUT=outputs/p122; CARD=0
run() { ./scripts/run_gaudi_dev.sh -m $TRAIN -m /mnt/cifar100 gaudi-env-cafl4ds:latest $CARD "$@"; }
BASE="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=mae stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
DEC='ssl.decoder_dim=512 ssl.decoder_depth=8 ssl.decoder_heads=16 ssl.decoder_mlp_ratio=4.0'
SEL="pool.kind=selection threads=8 fewshot.enabled=true fewshot.lr=3e-5 fewshot.ks=[100]"

drive() {  # name, well, overrides...
  name=$1; well=$2; shift 2
  [ -f $OUT/drive/$name/checkpoints/s0/final.pt ] && return
  run python scripts/run_deploy_harness.py $BASE "$SCENES" gate_arms=false eval_every=100 optim.lr=1e-4 stream.order=walk \
    well=$well checkpoint_legs=final 'seeds=[0]' study=P1.2.2 hydra.run.dir=$OUT/drive/$name "$@"
}

drive zero_lr wells/p121/mae_heads.pt optim.lr=0
drive lr_third wells/p121/mae_heads.pt optim.lr=3.3e-5
drive lr_x3 wells/p121/mae_heads.pt optim.lr=3e-4
drive warmup wells/p121/mae_heads.pt schedule=warmup_constant
drive protect wells/p121/mae_heads.pt param_groups.layer_decay=0.75 param_groups.no_decay=true

f=$OUT/encoders_sel.yaml
echo "- {name: b5, kind: well, path: wells/p121/mae_heads.pt}" > $f
echo "- {name: base, kind: checkpoint, path: outputs/p121/drive/mae_walk/checkpoints/s0/final.pt}" >> $f
for a in zero_lr lr_third lr_x3 warmup protect; do
  echo "- {name: $a, kind: checkpoint, path: $OUT/drive/$a/checkpoints/s0/final.pt}" >> $f
done
run python scripts/read_adaptation.py $BASE "$SCENES" $SEL '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/main

# The decoder arm: its own heads, settled on the frozen encoder exactly as P1.2.1's, then the drive.
[ -f wells/p122/mae_dec512_heads.pt ] || run python scripts/warm_well.py $BASE "$SCENES" $DEC optim.lr=3e-4 warmup.freeze_encoder=true \
  warmup.stop=loss warmup.window=200 warmup.tol=0.002 warmup.min_steps=500 warmup.max_steps=3000 \
  warmup.max_passes=100 warmup.select_every=500 well_out=wells/p122/mae_dec512_heads.pt hydra.run.dir=$OUT/wells/mae_dec512_heads
drive decoder wells/p122/mae_dec512_heads.pt $DEC
f=$OUT/encoders_sel_dec.yaml
echo "- {name: decoder, kind: checkpoint, path: $OUT/drive/decoder/checkpoints/s0/final.pt}" > $f
run python scripts/read_adaptation.py $BASE "$SCENES" $DEC $SEL '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/decoder
echo P122_ARMS_DONE
