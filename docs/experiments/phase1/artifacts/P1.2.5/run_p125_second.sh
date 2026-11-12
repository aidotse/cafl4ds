#!/bin/bash
# P1.2.5, the second batch — still independent of DINO's own objective (whose settings wait on the user).
# Card 0: Part A for DINO-init SimSiam — the three lr arms (walk, seed 0), read on the selection set, the pre-registered
#   choice (best end-of-drive scene linear, mean readout) written to select/choice.json, then its confirmation
#   (walk x seeds 1-5 with the curve's checkpoints, IID x seeds 1-3).
# Card 1: after the first batch's IID drive, the frozen floors on val (B5-DINO, the supervised ViT-B; reference reads),
#   then MAE-init's val ladder: its floor, the walk's checkpoints, the IID ends and Part D's controls.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; GEO=$TRAIN/geometry; OUT=outputs/p125
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/pr_2018_scaleout_workdir/cafl4ds/data/cifar100 -m /mnt/pr_2018_scaleout_workdir/cafl4ds/data \
  -m /home/mauricio/.cache/torch gaudi-env-cafl4ds:latest $card "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN \
 device=hpu img_size=224 stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=450 stream.select_query_per_canary=450"
PROTECT="gate_arms=false eval_every=25 param_groups.layer_decay=0.75 param_groups.no_decay=true study=P1.2.5"
# BDD val with box geometry (the FiftyOne-derived label file; frame-for-frame P1.2.1's pool)
GATE="data=bdd bdd_root=$BDD data.labels_file=$GEO/bdd100k_labels_images_val_box2d.json data.boxes=true \
 data.drivable_dir=$GEO/drivable/labels/val data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache \
 device=hpu img_size=224 pool.split=val pool.fit_fraction=0.4 threads=8 readout=mean cls_rungs=true"
TASKS='tasks=[scene,"presence_cell:traffic light",weather,"large_cell:car,bus,truck:0.02","presence_cell:bus","presence_cell:pedestrian"]'
FT='fewshot.tasks=[scene,"presence_cell:traffic light"]'

walk_list() {  # drive dir, first seed, n seeds -> walk_s<k>_<t> entries (few-shot at the end only)
  for i in $(seq 0 $(($3 - 1))); do
    d=$1/checkpoints/s$(($2 + i))
    for c in $(python3 -c "import json;print(' '.join(e['file'][:-3] for e in json.load(open('$d/index.json'))))"); do
      t=${c/step/t}; [ "$c" = final ] && t=end
      ft=false; [ "$t" = end ] && ft=true
      echo "- {name: walk_s${i}_$t, kind: checkpoint, path: $d/$c.pt, fewshot: $ft}"
    done
  done
}

card0() {
  W=wells/p125/dino_simsiam_heads.pt; B="encoder=vit_b16_dino backbone=je"
  for lr in 1e-5 3e-5 9e-5; do
    [ -f $OUT/drive/dino_simsiam_lr$lr/checkpoints/s0/final.pt ] || run 0 python scripts/run_deploy_harness.py $DIET "$SCENES" $B \
      $PROTECT optim.lr=$lr stream.order=walk well=$W checkpoint_legs=final 'seeds=[0]' hydra.run.dir=$OUT/drive/dino_simsiam_lr$lr
  done
  f=$OUT/select/dino_simsiam/encoders.yaml; mkdir -p $(dirname $f)
  echo "- {name: floor, kind: well, path: $W}" > $f
  for lr in 1e-5 3e-5 9e-5; do echo "- {name: lr$lr, kind: checkpoint, path: $OUT/drive/dino_simsiam_lr$lr/checkpoints/s0/final.pt}" >> $f; done
  [ -f $OUT/select/dino_simsiam/summary.json ] || run 0 python scripts/read_adaptation.py $DIET "$SCENES" $B pool.kind=selection \
    threads=8 readout=mean 'tasks=[scene]' '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/dino_simsiam
  python3 - $OUT/select/dino_simsiam <<'EOF'
import json, sys
s = json.load(open(f"{sys.argv[1]}/summary.json"))["encoders"]
reads = {n: e["bacc"]["scene|linear"]["pooled"] for n, e in s.items() if n.startswith("lr")}
json.dump({"reads": reads, "floor": s["floor"]["bacc"]["scene|linear"]["pooled"], "choice": max(reads, key=reads.get)},
          open(f"{sys.argv[1]}/choice.json", "w"), indent=2)
EOF
  lr=$(python3 -c "import json;print(json.load(open('$OUT/select/dino_simsiam/choice.json'))['choice'][2:])")
  [ -f $OUT/drive/dino_simsiam_walk/checkpoints/s5/final.pt ] || run 0 python scripts/run_deploy_harness.py $DIET "$SCENES" $B \
    $PROTECT optim.lr=$lr stream.order=walk well=$W checkpoint_legs=steps 'checkpoint_steps=[100,0.125,0.25,0.5,0.75]' \
    'seeds=[1,2,3,4,5]' hydra.run.dir=$OUT/drive/dino_simsiam_walk
  [ -f $OUT/drive/dino_simsiam_iid/checkpoints/s3/final.pt ] || run 0 python scripts/run_deploy_harness.py $DIET "$SCENES" $B \
    $PROTECT optim.lr=$lr stream.order=iid well=$W checkpoint_legs=final 'seeds=[1,2,3]' hydra.run.dir=$OUT/drive/dino_simsiam_iid
}

card1() {
  while [ ! -f $OUT/drive/mae_je_iid/checkpoints/s5/final.pt ] || pgrep -f "p125/drive/mae_je_iid" >/dev/null; do sleep 60; done
  for fl in dino:vit_b16_dino:1e-5 sup:vit_b16_pretrained:1e-4; do
    IFS=: read name enc lr <<< "$fl"
    [ -f $OUT/floors/$name/summary.json ] || run 1 python scripts/read_adaptation.py $GATE "$SCENES" encoder=$enc "$TASKS" "$FT" \
      fewshot.enabled=true fewshot.lr=$lr '~encoders' "+encoders=[{name:$name,kind:pretrained}]" hydra.run.dir=$OUT/floors/$name
  done
  f=$OUT/ladder/encoders_mae_je.yaml; mkdir -p $OUT/ladder
  echo "- {name: floor, kind: well, path: wells/p125/mae_je_heads.pt}" > $f
  walk_list $OUT/drive/mae_je_walk 1 5 >> $f
  for i in 0 1 2 3 4; do echo "- {name: iid_s${i}_end, kind: checkpoint, path: $OUT/drive/mae_je_iid/checkpoints/s$((i+1))/final.pt}" >> $f; done
  for i in 0 1 2; do echo "- {name: coco_s${i}_end, kind: checkpoint, path: $OUT/offdomain/drive/coco/checkpoints/s$i/final.pt}" >> $f; done
  for i in 0 1 2 3 4; do echo "- {name: places365_s${i}_end, kind: checkpoint, path: $OUT/offdomain/drive/places365/checkpoints/s$i/final.pt}" >> $f; done
  [ -f $OUT/ladder/mae_je/summary.json ] || run 1 python scripts/read_adaptation.py $GATE "$SCENES" encoder=vit_b16_mae backbone=je \
    "$TASKS" "$FT" fewshot.enabled=false fewshot.lr=3e-5 '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder/mae_je
}

card0 > $OUT/second_card0.log 2>&1 &
card1 > $OUT/second_card1.log 2>&1 &
wait
echo P125_SECOND_DONE
