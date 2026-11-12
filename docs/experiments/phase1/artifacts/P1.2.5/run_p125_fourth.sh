#!/bin/bash
# P1.2.5, Part G — the protections ablation, and every rung at every checkpoint (design: P1.2.5.md, Part G). One mode
# per call:
#   card0 (card 0): G1 — DINO-init SimSiam's selection grid (protections on x {1e-5,3e-5,9e-5}, off x {3e-6,1e-5,3e-5};
#          seeds 0, 6, 7 on the walk, ends only; the protected cells' seed 0 is Part A's drive), the selection read and
#          the pre-registered choice (best 3-seed mean scene linear) -> outputs/p125/partG/select/choice.json; the best
#          unprotected cell's confirmation (walk x seeds 1-5, Part B's checkpoints); if the chosen cell is not Part A's
#          (protections, 3e-5), its confirmation and Part D's controls under it; that arm's val ladders (every rung at
#          every checkpoint; F1's size tasks on the frozen rungs); then G3's re-read of MAE-init JE.
#   card1 (card 1): G3's re-read of DINO-init DINO and DINO-init SimSiam (the fine-tune rungs at every checkpoint), then
#          G2 — MAE-init JE without the protections under P1.2.5's conditions: the walk x seeds 1-5 with the checkpoints,
#          BDD IID x 5, COCO x 3, Places365 x 5, and its val ladders.
#   dinoft (card 1): DINO-init DINO's fine-tune rungs re-read after the frozen-teacher fix (see the function).
#   analyze (host): analyze_adaptation mode=p125g -> outputs/p125/partG.json.
# "Without the protections" is P1.2.1's recipe: no layer-wise decay, weight decay on every parameter (the harness's
# defaults). Drives write to the project share, linked from outputs/p125/partG.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
BDD=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; GEO=$BDD/geometry; OUT=outputs/p125
SHARE=/mnt/pr_2018_scaleout_workdir/cafl4ds; SH=$SHARE/outputs/p125
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /home/mauricio/.cache/torch -m $SHARE:$SHARE \
  gaudi-env-cafl4ds:latest $card "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$BDD \
 device=hpu img_size=224 stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=450 stream.select_query_per_canary=450"
PROTECT="gate_arms=false eval_every=25 param_groups.layer_decay=0.75 param_groups.no_decay=true study=P1.2.5"
NOPROT="gate_arms=false eval_every=25 study=P1.2.5"
GATE="data=bdd bdd_root=$BDD data.labels_file=$GEO/bdd100k_labels_images_val_box2d.json data.boxes=true \
 data.drivable_dir=$GEO/drivable/labels/val data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache \
 device=hpu img_size=224 pool.split=val pool.fit_fraction=0.4 threads=8 readout=mean"
TASKS='tasks=[scene,"presence_cell:traffic light",weather,"large_cell:car,bus,truck:0.02","presence_cell:bus","presence_cell:pedestrian"]'
GATETASKS='tasks=[scene,"presence_cell:traffic light"]'
FT='fewshot.tasks=[scene,"presence_cell:traffic light"]'
SIM="encoder=vit_b16_dino backbone=je"; MAE="encoder=vit_b16_mae backbone=je"
WS=wells/p125/dino_simsiam_heads.pt; WM=wells/p125/mae_je_heads.pt
SUB=0.005102; LARGE=0.02
F1TASKS='tasks=['
for c in "traffic sign" pedestrian truck bus bicycle rider motorcycle; do
  F1TASKS+="\"presence_cell:$c\",\"size_cell:$c:0:$SUB\",\"size_cell:$c:$SUB:$LARGE\",\"size_cell:$c:$LARGE:inf\","
done
F1TASKS="${F1TASKS%,}]"

link() {  # a run dir on the share, linked from outputs/p125
  mkdir -p $SH/$1 $(dirname $OUT/$1); [ -e $OUT/$1 ] || ln -s $SH/$1 $OUT/$1
}

walk_list() {  # drive dir, first seed, n seeds -> walk_s<k>_<t> entries, the fine-tune rungs at every checkpoint
  for i in $(seq 0 $(($3 - 1))); do
    d=$1/checkpoints/s$(($2 + i))
    for c in $(python3 -c "import json;print(' '.join(e['file'][:-3] for e in json.load(open('$d/index.json'))))"); do
      t=${c/step/t}; [ "$c" = final ] && t=end
      echo "- {name: walk_s${i}_$t, kind: checkpoint, path: $d/$c.pt, fewshot: true}"
    done
  done
}

same_query() {  # two ladders read the same val query frames
  cmp -s <(uv run python -c "import numpy as n;print(n.load('$1/pool.npz')['query_index'].tolist())") \
    <(uv run python -c "import numpy as n;print(n.load('$2/pool.npz')['query_index'].tolist())") \
    || { echo "QUERY MISMATCH $1 $2"; return 1; }
}

drive() {  # card dir recipe-flags...: one drive set on the share, skipped once its last seed's end exists
  card=$1; dir=$2; last=$3; shift 3
  link $dir
  [ -f $OUT/$dir/checkpoints/$last/final.pt ] || run $card python scripts/run_deploy_harness.py "$@" hydra.run.dir=$SH/$dir
}

controls() {  # card arm well encoder-flags recipe-flags: Part D's set (BDD IID x 5, COCO x 3, Places365 x 5)
  card=$1; arm=$2; well=$3; enc=$4; recipe=$5
  CTRL="img_size=224 device=hpu $enc data.cache_dir=/workspace/outputs/cache stream.order=iid \
   stream.support_per_canary=10 stream.query_per_canary=10 stream.stratify_canary=false resume_check_split=false \
   data.subset=54184 data.subset_seed=0"
  drive $card partG/drive/${arm}_iid s5 $DIET "$SCENES" $enc $recipe stream.order=iid well=$well checkpoint_legs=final \
    'seeds=[1,2,3,4,5]'
  for sc in coco:'[0,1,2]':s2 places365:'[0,1,2,3,4]':s4; do
    IFS=: read src seeds last <<< "$sc"
    drive $card partG/offdomain/$arm/$src $last data=$src $CTRL $recipe well=$well checkpoint_legs=final "seeds=$seeds"
  done
}

ladders() {  # card arm floor-entry ft-lr walk-dir with-controls encoder-flags...: the arm's val ladder (every rung at
  # every checkpoint), then F1's size tasks on the frozen rungs
  card=$1; arm=$2; floor=$3; lr=$4; walk=$5; ctrl=$6; shift 6
  f=$OUT/ladder/encoders_$arm.yaml
  echo "$floor" > $f
  walk_list $OUT/$walk 1 5 >> $f
  if [ "$ctrl" = yes ]; then
    for i in 0 1 2 3 4; do echo "- {name: iid_s${i}_end, kind: checkpoint, path: $OUT/partG/drive/${arm}_iid/checkpoints/s$((i+1))/final.pt, fewshot: false}" >> $f; done
    for i in 0 1 2; do echo "- {name: coco_s${i}_end, kind: checkpoint, path: $OUT/partG/offdomain/$arm/coco/checkpoints/s$i/final.pt, fewshot: false}" >> $f; done
    for i in 0 1 2 3 4; do echo "- {name: places365_s${i}_end, kind: checkpoint, path: $OUT/partG/offdomain/$arm/places365/checkpoints/s$i/final.pt, fewshot: false}" >> $f; done
  fi
  [ -f $OUT/ladder/$arm/summary.json ] || run $card python scripts/read_adaptation.py $GATE cls_rungs=true "$SCENES" "$@" \
    "$TASKS" "$FT" fewshot.enabled=true fewshot.lr=$lr '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder/$arm
  same_query $OUT/ladder/dino_simsiam $OUT/ladder/$arm
  g=$OUT/ladderF/encoders_$arm.yaml
  grep -E "name: (floor|walk_s)" $f | sed 's/fewshot: true/fewshot: false/' > $g
  [ -f $OUT/ladderF/$arm/summary.json ] || run $card python scripts/read_adaptation.py $GATE "$SCENES" "$@" "$F1TASKS" \
    cls_rungs=false fewshot.enabled=false 'fewshot.tasks=[]' '~encoders' encoders_file=$g hydra.run.dir=$OUT/ladderF/$arm
  same_query $OUT/ladderF/dino_simsiam $OUT/ladderF/$arm
}

reread() {  # card arm ft-lr encoder-flags...: G3 — a confirmation drive's floor and checkpoints on the gate tasks,
  # every rung at every checkpoint (its ends again, as a check)
  card=$1; arm=$2; lr=$3; shift 3
  f=$OUT/ladder_ft/encoders_$arm.yaml; mkdir -p $OUT/ladder_ft
  grep -E "name: (floor|walk_s)" $OUT/ladder/encoders_$arm.yaml | sed 's/fewshot: false/fewshot: true/' > $f
  [ -f $OUT/ladder_ft/$arm/summary.json ] || run $card python scripts/read_adaptation.py $GATE cls_rungs=false "$SCENES" \
    "$@" "$GATETASKS" "$FT" fewshot.enabled=true fewshot.lr=$lr '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder_ft/$arm
  same_query $OUT/ladder/$arm $OUT/ladder_ft/$arm
}

card0() {
  for lr in 3e-6 1e-5 3e-5; do
    drive 0 partG/sel/noprot_lr$lr s7 $DIET "$SCENES" $SIM $NOPROT optim.lr=$lr stream.order=walk well=$WS \
      checkpoint_legs=final 'seeds=[0,6,7]'
  done
  for lr in 1e-5 3e-5 9e-5; do
    drive 0 partG/sel/prot_lr$lr s7 $DIET "$SCENES" $SIM $PROTECT optim.lr=$lr stream.order=walk well=$WS \
      checkpoint_legs=final 'seeds=[6,7]'
  done
  f=$OUT/partG/select/encoders.yaml; mkdir -p $OUT/partG/select
  echo "- {name: floor, kind: well, path: $WS}" > $f
  for lr in 1e-5 3e-5 9e-5; do
    echo "- {name: prot_lr${lr}_s0, kind: checkpoint, path: $OUT/drive/dino_simsiam_lr$lr/checkpoints/s0/final.pt}" >> $f
    for s in 6 7; do echo "- {name: prot_lr${lr}_s$s, kind: checkpoint, path: $OUT/partG/sel/prot_lr$lr/checkpoints/s$s/final.pt}" >> $f; done
  done
  for lr in 3e-6 1e-5 3e-5; do
    for s in 0 6 7; do echo "- {name: noprot_lr${lr}_s$s, kind: checkpoint, path: $OUT/partG/sel/noprot_lr$lr/checkpoints/s$s/final.pt}" >> $f; done
  done
  [ -f $OUT/partG/select/summary.json ] || run 0 python scripts/read_adaptation.py $DIET "$SCENES" $SIM pool.kind=selection \
    threads=8 readout=mean 'tasks=[scene]' '~encoders' encoders_file=$f hydra.run.dir=$OUT/partG/select
  python3 - $OUT/partG/select <<'EOF'
import json, re, sys
from collections import defaultdict
s = json.load(open(f"{sys.argv[1]}/summary.json"))["encoders"]
seeds = defaultdict(dict)
for name, e in s.items():
    if m := re.fullmatch(r"(\w+_lr[0-9e.-]+)_s(\d+)", name):
        seeds[m.group(1)][m.group(2)] = e["bacc"]["scene|linear"]["pooled"]
means = {cell: sum(r.values()) / len(r) for cell, r in seeds.items()}
unprotected = {c: m for c, m in means.items() if c.startswith("noprot")}
json.dump({"reads": seeds, "means": means, "floor": s["floor"]["bacc"]["scene|linear"]["pooled"],
           "choice": max(means, key=means.get), "best_unprotected": max(unprotected, key=unprotected.get)},
          open(f"{sys.argv[1]}/choice.json", "w"), indent=2)
EOF
  cat $OUT/partG/select/choice.json
  choice=$(python3 -c "import json;print(json.load(open('$OUT/partG/select/choice.json'))['choice'])")
  best=$(python3 -c "import json;print(json.load(open('$OUT/partG/select/choice.json'))['best_unprotected'])")
  drive 0 partG/drive/dino_simsiam_np_walk s5 $DIET "$SCENES" $SIM $NOPROT optim.lr=${best#noprot_lr} stream.order=walk \
    well=$WS checkpoint_legs=steps \
    'checkpoint_steps=[100,0.125,0.25,0.5,0.75]' 'seeds=[1,2,3,4,5]'
  if [ "$choice" = "$best" ]; then
    controls 0 dino_simsiam_np $WS "$SIM" "$NOPROT optim.lr=${best#noprot_lr}"
    ladders 0 dino_simsiam_np "- {name: floor, kind: well, path: $WS}" 1e-5 partG/drive/dino_simsiam_np_walk yes $SIM
  else
    ladders 0 dino_simsiam_np "- {name: floor, kind: well, path: $WS}" 1e-5 partG/drive/dino_simsiam_np_walk no $SIM
    if [ "$choice" != prot_lr3e-5 ]; then  # a protected cell other than Part A's: its confirmation and controls
      arm=dino_simsiam_p${choice#prot_lr}
      drive 0 partG/drive/${arm}_walk s5 $DIET "$SCENES" $SIM $PROTECT optim.lr=${choice#prot_lr} stream.order=walk \
        well=$WS checkpoint_legs=steps \
    'checkpoint_steps=[100,0.125,0.25,0.5,0.75]' 'seeds=[1,2,3,4,5]'
      controls 0 $arm $WS "$SIM" "$PROTECT optim.lr=${choice#prot_lr}"
      ladders 0 $arm "- {name: floor, kind: well, path: $WS}" 1e-5 partG/drive/${arm}_walk yes $SIM
    fi
  fi
  reread 0 mae_je 3e-5 $MAE
}

card1() {
  reread 1 dino_simsiam 1e-5 $SIM
  reread 1 dino_dino 1e-5 encoder=vit_b16_dino backbone=dino
  drive 1 partG/drive/mae_je_np_walk s5 $DIET "$SCENES" $MAE $NOPROT optim.lr=3e-5 stream.order=walk well=$WM checkpoint_legs=steps \
    'checkpoint_steps=[100,0.125,0.25,0.5,0.75]' \
    'seeds=[1,2,3,4,5]'
  controls 1 mae_je_np $WM "$MAE" "$NOPROT optim.lr=3e-5"
  ladders 1 mae_je_np "- {name: floor, kind: well, path: $WM}" 3e-5 partG/drive/mae_je_np_walk yes $MAE
}

dinoft() {  # card 1: DINO-init DINO's fine-tune rungs again, after the fix that trains the encoder in every fine-tune
  # (a DINO checkpoint's teacher is loaded with its grads off, so the first read trained the head alone); the first
  # read is kept beside it as ladder_ft/dino_dino_headonly
  [ -d $OUT/ladder_ft/dino_dino_headonly ] || mv $OUT/ladder_ft/dino_dino $OUT/ladder_ft/dino_dino_headonly
  reread 1 dino_dino 1e-5 encoder=vit_b16_dino backbone=dino
  # the corrected fine-tune rungs replace the walk ends' in ladder/dino_dino (the first read kept in ladder_headonly/);
  # the floor and every frozen rung must match exactly. Then re-run analyze_adaptation mode=p125.
  uv run python - "$OUT" <<'EOF'
import json, shutil, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
src, fix, keep = root / "ladder/dino_dino", root / "ladder_ft/dino_dino", root / "ladder_headonly/dino_dino"
if not keep.exists():
    shutil.copytree(src, keep, symlinks=True)
summary = json.loads((keep / "summary.json").read_text())
fixed = json.loads((fix / "summary.json").read_text())["encoders"]
for name in ["floor", *[f"walk_s{s}_end" for s in range(5)]]:
    old, new = dict(np.load(keep / f"{name}.npz")), np.load(fix / f"{name}.npz")
    for key in new.files:
        if "|ft" in key and name != "floor":
            old[key] = new[key]
            summary["encoders"][name]["bacc"][key] = fixed[name]["bacc"][key]
        else:
            assert np.array_equal(old[key], new[key]), (name, key)
    np.savez(src / f"{name}.npz", **old)
(src / "summary.json").write_text(json.dumps(summary, indent=2))
EOF
}

analyze() {  # host: every Part G ladder present
  arms=$(cd $OUT/ladder && ls -d dino_simsiam_np dino_simsiam_p* mae_je_np 2>/dev/null | paste -sd, -)
  att=$(for a in ${arms//,/ }; do [ -e $OUT/ladder/$a/coco_s0_end.npz ] && printf '%s:{coco:3,places365:5},' $a; done)
  pairs=$(for a in ${arms//,/ }; do case $a in mae_je_np) printf 'mae_je_np:mae_je,';; dino_simsiam_np) printf 'dino_simsiam_np:dino_simsiam,';; esac; done)
  ft=$(cd $OUT/ladder_ft && ls -d dino_dino dino_simsiam mae_je 2>/dev/null | paste -sd, -)
  uv run python scripts/analyze_adaptation.py mode=p125g root=$OUT n_iid_seeds=5 report=partG.json \
    'p125.panel=[weather,"large_cell:car,bus,truck:0.02","presence_cell:bus","presence_cell:pedestrian"]' \
    "p125g.arms=[$arms]" "+p125g.attribution={${att%,}}" "+p125g.pairs={${pairs%,}}" "p125g.ft_arms=[$ft]" \
    hydra.run.dir=$OUT/analysis_partG
}

# One braced block ending in exit: bash parses it whole before running, so editing this file mid-run is safe.
{ "$1" > $OUT/fourth_$1.log 2>&1; echo "P125_FOURTH_$1_DONE" >> $OUT/fourth_$1.log; exit; }
