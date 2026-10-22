#!/bin/bash
# P1.2.4 — the BDD-only pieces, card 1: A2's frozen floors on P1.2.1's val ladder; B1's interpolation, B2-B3's one-seed
# JE arms (walk, seed 0, from P1.2.1's settled-head JE well), all chosen on the train-side selection set; then P1.2.2's
# per-block read on P1.2.1's checkpoints (val, frozen rungs; a diagnosis, never a choice).
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/home/mauricio/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p124; CARD=1
run() { ./scripts/run_gaudi_dev.sh -m $TRAIN -m $BDD -m /mnt/cifar100 -m /home/mauricio/data/weights -m /home/mauricio/.cache/torch \
  gaudi-env-cafl4ds:latest $CARD "$@"; }
BASE="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN device=hpu img_size=224 \
 encoder=vit_b16_mae backbone=je stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=150 stream.select_query_per_canary=150"
SEL="pool.kind=selection threads=8"
GATE="data=bdd bdd_root=$BDD data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 \
 pool.split=val pool.fit_fraction=0.4 threads=8"
TASKS='tasks=[scene,"presence_cell:traffic light"]'

# A2: the frozen floors, on P1.2.1's val ladder exactly (same pool, tasks, rungs; B5's reads are P1.2.1's).
for fl in dino:vit_b16_dino sup:vit_b16_pretrained; do
  name=${fl%%:*}; enc=${fl#*:}
  [ -f $OUT/floors/$name/summary.json ] || run python scripts/read_adaptation.py $GATE "$SCENES" encoder=$enc "$TASKS" \
    fewshot.enabled=true fewshot.lr=3e-5 '~encoders' "+encoders=[{name:$name,kind:pretrained}]" hydra.run.dir=$OUT/floors/$name
done

# B1: interpolation between B5 and each P1.2.1 walk-end checkpoint, on the selection set (frozen rungs).
f=$OUT/encoders_interp.yaml; W=wells/p121/je_heads.pt
echo "- {name: b5, kind: well, path: $W}" > $f
for s in 0 1 2 3 4; do
  c=outputs/p121/drive/je_walk/checkpoints/s$s/final.pt
  echo "- {name: walk_s${s}_a1.0, kind: checkpoint, path: $c}" >> $f
  for a in 0.25 0.5 0.75; do
    echo "- {name: walk_s${s}_a$a, kind: interp, alpha: $a, a: {kind: well, path: $W}, b: {kind: checkpoint, path: $c}}" >> $f
  done
done
[ -f $OUT/select/interp/summary.json ] || run python scripts/read_adaptation.py $BASE "$SCENES" $SEL '~encoders' \
  encoders_file=$f hydra.run.dir=$OUT/select/interp

# B2-B3: one-seed recipe arms.
drive() {  # name, overrides...
  name=$1; shift
  [ -f $OUT/drive/$name/checkpoints/s0/final.pt ] && return
  run python scripts/run_deploy_harness.py $BASE "$SCENES" gate_arms=false eval_every=100 optim.lr=3e-5 stream.order=walk \
    well=$W checkpoint_legs=final 'seeds=[0]' study=P1.2.4 hydra.run.dir=$OUT/drive/$name "$@"
}
drive cosine schedule=warmup_cosine
drive ema ema=0.999
drive lr_third optim.lr=1e-5
drive lr_x3 optim.lr=9e-5
drive protect param_groups.layer_decay=0.75 param_groups.no_decay=true

f=$OUT/encoders_sel.yaml
echo "- {name: b5, kind: well, path: $W}" > $f
echo "- {name: base, kind: checkpoint, path: outputs/p121/drive/je_walk/checkpoints/s0/final.pt}" >> $f
for a in cosine ema lr_third lr_x3 protect; do
  echo "- {name: $a, kind: checkpoint, path: $OUT/drive/$a/checkpoints/s0/final.pt}" >> $f
done
echo "- {name: ema_avg, kind: checkpoint, path: $OUT/drive/ema/checkpoints/s0/ema_final.pt}" >> $f
run python scripts/read_adaptation.py $BASE "$SCENES" $SEL '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/arms

# P1.2.2's per-block read: B5 and P1.2.1's five walk-end checkpoints per backbone, blocks 4-12, frozen rungs on val.
for bb in mae je; do
  f=outputs/p122/encoders_blocks_$bb.yaml
  echo "- {name: b5, kind: well, path: wells/p121/${bb}_heads.pt}" > $f
  for s in 0 1 2 3 4; do echo "- {name: walk_s${s}_end, kind: checkpoint, path: outputs/p121/drive/${bb}_walk/checkpoints/s$s/final.pt}" >> $f; done
  [ -f outputs/p122/blocks/$bb/summary.json ] || run python scripts/read_adaptation.py $GATE "$SCENES" encoder=vit_b16_mae backbone=$bb "$TASKS" \
    'blocks=[4,5,6,7,8,9,10,11,12]' '~encoders' encoders_file=$f hydra.run.dir=outputs/p122/blocks/$bb
done
echo P124_BDD_DONE
