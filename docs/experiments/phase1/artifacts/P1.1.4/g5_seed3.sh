#!/bin/bash
# P1.1 audit G5: the fine-tune read P1.1.4's `degradation` label could not make on DINO-init SimSiam's IID seed 3
# (P1.2.5's drive, end state only; its kNN canary fires at step 1,175). The same val ladder as run_p114.sh's
# `reads_je` (gate tasks, kNN, linear, 10- and 100-shot fine-tunes at the floor's rate 1e-5), named by drive seed.
# Usage: g5_seed3.sh <card>
set -u
cd /home/mauricio/projects/cafl4ds
card=${1:-0}
SCENES='data.scenes=["city street",highway,residential]'
BDD=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; OUT=outputs/p114
SHARE=/mnt/pr_2018_scaleout_workdir/cafl4ds; SH=$SHARE/outputs/p114
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /home/mauricio/.cache/torch -m $SHARE:$SHARE \
  gaudi-env-cafl4ds:latest $card "$@"; }
GATE="data=bdd bdd_root=$BDD data.labels_file=$BDD/geometry/bdd100k_labels_images_val_box2d.json data.boxes=true \
 data.drivable_dir=$BDD/geometry/drivable/labels/val data.min_canary_count=50 data.pixel_dtype=uint8 \
 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 pool.split=val pool.fit_fraction=0.4 threads=8 readout=mean"
GATETASKS='tasks=[scene,"presence_cell:traffic light"]'; FT='fewshot.tasks=[scene,"presence_cell:traffic light"]'
SIM="encoder=vit_b16_dino backbone=je"; WS=wells/p125/dino_simsiam_heads.pt
name=dino_simsiam_iid_p125
mkdir -p $SH/ladder/$name
f=$SH/ladder/encoders_$name.yaml
cat > $f <<EOF
- {name: floor, kind: well, path: $WS, fewshot: true}
- {name: iid_s3_end, kind: checkpoint, path: outputs/p125/partG/drive/dino_simsiam_p9e-5_iid/checkpoints/s3/final.pt, fewshot: true}
EOF
[ -f $OUT/ladder/$name/summary.json ] || run $card python scripts/read_adaptation.py $GATE cls_rungs=false "$SCENES" \
  $SIM "$GATETASKS" "$FT" fewshot.enabled=true fewshot.lr=1e-5 '~encoders' encoders_file=$f hydra.run.dir=$SH/ladder/$name
cmp -s <(uv run python -c "import numpy as n;print(n.load('outputs/p125/ladder/dino_simsiam/pool.npz')['query_index'].tolist())") \
  <(uv run python -c "import numpy as n;print(n.load('$OUT/ladder/$name/pool.npz')['query_index'].tolist())") \
  || echo "QUERY MISMATCH $name"
echo G5_DONE
