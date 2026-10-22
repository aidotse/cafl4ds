#!/bin/bash
# P1.2.2's seventh arm: the two levers that helped on the selection set (512x8 decoder + protections), card 0.
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
[ -f $OUT/drive/decoder_protect/checkpoints/s0/final.pt ] || run python scripts/run_deploy_harness.py $BASE "$SCENES" $DEC gate_arms=false eval_every=100 \
  optim.lr=1e-4 stream.order=walk well=wells/p122/mae_dec512_heads.pt checkpoint_legs=final 'seeds=[0]' study=P1.2.2 \
  param_groups.layer_decay=0.75 param_groups.no_decay=true hydra.run.dir=$OUT/drive/decoder_protect
f=$OUT/encoders_sel_combo.yaml
echo "- {name: decoder_protect, kind: checkpoint, path: $OUT/drive/decoder_protect/checkpoints/s0/final.pt}" > $f
run python scripts/read_adaptation.py $BASE "$SCENES" $DEC $SEL '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/combo
echo P122_COMBO_DONE
