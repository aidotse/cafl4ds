#!/bin/bash
# P1.2.5, the third batch — DINO's own objective, and the val reads. One mode per call:
#   partE  (card 1, after the first batch): Part E's collapse control — the first 300 steps of DINO-init DINO's walk at
#          lr 3e-5 (protections recipe), unswitched and with each collapse switch. Read before `dino` starts.
#   dinolr (card 1): only the three lr arms of `dino` (they run while Part E's read is open).
#   partEfull (card 1, after dinolr): the selection read and choice, then each switch over the full drive at the
#          chosen lr (that lr's arm is the unswitched drive). Read before `dino`'s confirmation.
#   dinowalk / dinoiid (cards 1 / 0): the confirmation's drives alone, at the chosen lr (they run while Part E's
#          read is open; nothing is read from them).
#   partEhi (card 1, after dinowalk): Part E's positive control at lr 3e-4 and 1e-3 (600 steps, twin + each switch).
#   dino   (card 0 for its read; drives skip once done): Part A for DINO-init DINO — the three lr arms, the selection read and pre-registered choice, the
#          confirmation (walk x seeds 1-5 with the curve's checkpoints, IID x seeds 1-3), then its val ladder.
#   simsiamctrl (card 0, once free): Part D's controls for DINO-init SimSiam, and their val read, joined to its ladder.
#   analyze (host): analyze_adaptation mode=p125 over every ladder present -> outputs/p125/adaptation_report.json.
#   partFa / partFb (cards 0 / 1): Part F1's size-binned presence reads -> outputs/p125/ladderF/<arm> (partFb waits
#          for partFa's pixel cache). partFc (both cards, after them): F1 on P1.2.1's arms -> outputs/p121/ladderF/.
#   reads  (card 0, after the second batch): the frozen floors on val (B5-DINO, the supervised ViT-B), then the val
#          ladders of MAE-init JE and DINO-init SimSiam.
# DINO's drives write to the NFS project share (a DINO checkpoint is ~0.85 GB; the root disk is short), linked from
# outputs/p125 so every path below reads the same.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
TRAIN=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; BDD=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; GEO=$TRAIN/geometry; OUT=outputs/p125
SHARE=/mnt/pr_2018_scaleout_workdir/cafl4ds; SH=$SHARE/outputs/p125
# the weights and every dataset live on the share's data/ (BDD since 2026-11-06; $TRAIN and $BDD are one tree)
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /home/mauricio/.cache/torch -m $SHARE:$SHARE \
  gaudi-env-cafl4ds:latest $card "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$TRAIN \
 device=hpu img_size=224 stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=450 stream.select_query_per_canary=450"
PROTECT="gate_arms=false eval_every=25 param_groups.layer_decay=0.75 param_groups.no_decay=true study=P1.2.5"
GATE="data=bdd bdd_root=$BDD data.labels_file=$GEO/bdd100k_labels_images_val_box2d.json data.boxes=true \
 data.drivable_dir=$GEO/drivable/labels/val data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache \
 device=hpu img_size=224 pool.split=val pool.fit_fraction=0.4 threads=8 readout=mean cls_rungs=true"
TASKS='tasks=[scene,"presence_cell:traffic light",weather,"large_cell:car,bus,truck:0.02","presence_cell:bus","presence_cell:pedestrian"]'
FT='fewshot.tasks=[scene,"presence_cell:traffic light"]'
DINO="encoder=vit_b16_dino backbone=dino"

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

link() {  # a run dir on the share, linked from outputs/p125
  mkdir -p $SH/$1 $(dirname $OUT/$1); [ -e $OUT/$1 ] || ln -s $SH/$1 $OUT/$1
}

partE() {
  until grep -q P125_FIRST_DONE $OUT/first.log; do sleep 60; done
  for sw in dino: nocenter:ssl.centering=false nosharpen:ssl.sharpening=false; do
    IFS=: read name flag <<< "$sw"; link partE/$name
    [ -f $OUT/partE/$name/checkpoints/s0/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET "$SCENES" $DINO \
      $PROTECT optim.lr=3e-5 stream.order=walk max_steps=300 checkpoint_legs=final 'seeds=[0]' $flag \
      hydra.run.dir=$SH/partE/$name
  done
}

dinolr() {  # the three lr arms alone (needed whatever Part E's read decides; nothing here is read as healthy)
  for lr in 1e-5 3e-5 9e-5; do
    link drive/dino_dino_lr$lr
    [ -f $OUT/drive/dino_dino_lr$lr/checkpoints/s0/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET "$SCENES" \
      $DINO $PROTECT optim.lr=$lr stream.order=walk checkpoint_legs=final 'seeds=[0]' hydra.run.dir=$SH/drive/dino_dino_lr$lr
  done
}

partEhi() {  # Part E's positive control (the user's option B, 2026-10-28): does DINO's collapse reach the backbone
  # rank and the canary at a higher lr? Per lr, an unswitched twin and each switch, 600 steps, read twin against arm (as
  # P1.1.1's collapse dial); card 1 after the confirmation's walk
  until grep -q P125_THIRD_dinowalk_DONE $OUT/third_dinowalk.log; do sleep 60; done
  for lr in 3e-4 1e-3; do
    for sw in dino: nocenter:ssl.centering=false nosharpen:ssl.sharpening=false; do
      IFS=: read name flag <<< "$sw"; link partE/hi/${name}_lr$lr
      [ -f $OUT/partE/hi/${name}_lr$lr/checkpoints/s0/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET \
        "$SCENES" $DINO $PROTECT optim.lr=$lr stream.order=walk max_steps=600 checkpoint_legs=final 'seeds=[0]' $flag \
        hydra.run.dir=$SH/partE/hi/${name}_lr$lr
    done
  done
}

dinoselect() {  # the selection read of the lr arms, and the pre-registered choice
  f=$OUT/select/dino_dino/encoders.yaml; mkdir -p $(dirname $f)
  echo "- {name: floor, kind: pretrained}" > $f
  for lr in 1e-5 3e-5 9e-5; do echo "- {name: lr$lr, kind: checkpoint, path: $OUT/drive/dino_dino_lr$lr/checkpoints/s0/final.pt}" >> $f; done
  [ -f $OUT/select/dino_dino/summary.json ] || run 1 python scripts/read_adaptation.py $DIET "$SCENES" $DINO pool.kind=selection \
    threads=8 readout=mean 'tasks=[scene]' '~encoders' encoders_file=$f hydra.run.dir=$OUT/select/dino_dino
  python3 - $OUT/select/dino_dino <<'EOF'
import json, sys
s = json.load(open(f"{sys.argv[1]}/summary.json"))["encoders"]
reads = {n: e["bacc"]["scene|linear"]["pooled"] for n, e in s.items() if n.startswith("lr")}
json.dump({"reads": reads, "floor": s["floor"]["bacc"]["scene|linear"]["pooled"], "choice": max(reads, key=reads.get)},
          open(f"{sys.argv[1]}/choice.json", "w"), indent=2)
EOF
}

partEfull() {  # Part E lengthened (the user's call, 2026-10-28): each switch over the full drive at the chosen lr,
  # against that lr's arm as the unswitched drive
  while pgrep -f "gaudi-env-cafl4ds:latest 1 " >/dev/null; do sleep 60; done  # card 1 free (an lr arm may be running)
  dinolr; dinoselect
  lr=$(python3 -c "import json;print(json.load(open('$OUT/select/dino_dino/choice.json'))['choice'][2:])")
  for sw in nocenter:ssl.centering=false nosharpen:ssl.sharpening=false; do
    IFS=: read name flag <<< "$sw"; link partE/${name}_full
    [ -f $OUT/partE/${name}_full/checkpoints/s0/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET "$SCENES" \
      $DINO $PROTECT optim.lr=$lr stream.order=walk checkpoint_legs=final 'seeds=[0]' $flag hydra.run.dir=$SH/partE/${name}_full
  done
}

dinowalk() {  # the confirmation's walk (card 1); drives only, nothing read
  lr=$(python3 -c "import json;print(json.load(open('$OUT/select/dino_dino/choice.json'))['choice'][2:])")
  link drive/dino_dino_walk
  [ -f $OUT/drive/dino_dino_walk/checkpoints/s5/final.pt ] || run 1 python scripts/run_deploy_harness.py $DIET "$SCENES" \
    $DINO $PROTECT optim.lr=$lr stream.order=walk checkpoint_legs=steps 'checkpoint_steps=[100,0.125,0.25,0.5,0.75]' \
    'seeds=[1,2,3,4,5]' hydra.run.dir=$SH/drive/dino_dino_walk
}

dinoiid() {  # the confirmation's IID control (card 0); drives only, nothing read
  lr=$(python3 -c "import json;print(json.load(open('$OUT/select/dino_dino/choice.json'))['choice'][2:])")
  link drive/dino_dino_iid
  [ -f $OUT/drive/dino_dino_iid/checkpoints/s3/final.pt ] || run 0 python scripts/run_deploy_harness.py $DIET "$SCENES" \
    $DINO $PROTECT optim.lr=$lr stream.order=iid checkpoint_legs=final 'seeds=[1,2,3]' hydra.run.dir=$SH/drive/dino_dino_iid
}

dino() {  # the val ladder of DINO-init DINO — only once Part E is settled (the drives above run before)
  dinolr; dinoselect; dinowalk; dinoiid
  f=$OUT/ladder/encoders_dino_dino.yaml; mkdir -p $OUT/ladder
  echo "- {name: floor, kind: pretrained}" > $f
  walk_list $SH/drive/dino_dino_walk 1 5 >> $f
  for i in 0 1 2; do echo "- {name: iid_s${i}_end, kind: checkpoint, path: $SH/drive/dino_dino_iid/checkpoints/s$((i+1))/final.pt, fewshot: false}" >> $f; done
  [ -f $OUT/ladder/dino_dino/summary.json ] || run 0 python scripts/read_adaptation.py $GATE "$SCENES" $DINO "$TASKS" "$FT" \
    fewshot.enabled=true fewshot.lr=1e-5 '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder/dino_dino
}

simsiamctrl() {  # Part D for DINO-init SimSiam (it passed its gate, 2026-10-28): COCO x 3, Places365 x 5 from its well,
  # the BDD IID arm topped up to 5 seeds, then their val read; card 0 once it is free
  while pgrep -f "gaudi-env-cafl4ds:latest 0 " >/dev/null; do sleep 60; done
  W=wells/p125/dino_simsiam_heads.pt
  CTRL="img_size=224 device=hpu encoder=vit_b16_dino backbone=je data.cache_dir=/workspace/outputs/cache stream.order=iid \
   stream.support_per_canary=10 stream.query_per_canary=10 stream.stratify_canary=false resume_check_split=false \
   data.subset=54184 data.subset_seed=0"
  for sc in coco:'[0,1,2]':s2 places365:'[0,1,2,3,4]':s4; do
    IFS=: read src seeds last <<< "$sc"; link offdomain/dino_simsiam/$src
    [ -f $OUT/offdomain/dino_simsiam/$src/checkpoints/$last/final.pt ] || run 0 python scripts/run_deploy_harness.py \
      data=$src $CTRL $PROTECT optim.lr=3e-5 well=$W checkpoint_legs=final "seeds=$seeds" \
      hydra.run.dir=$SH/offdomain/dino_simsiam/$src
  done
  [ -f $OUT/drive/dino_simsiam_iid45/checkpoints/s5/final.pt ] || run 0 python scripts/run_deploy_harness.py $DIET "$SCENES" \
    encoder=vit_b16_dino backbone=je $PROTECT optim.lr=3e-5 stream.order=iid well=$W checkpoint_legs=final 'seeds=[4,5]' \
    hydra.run.dir=$OUT/drive/dino_simsiam_iid45
  f=$OUT/ladder/encoders_dino_simsiam_controls.yaml
  : > $f
  for i in 3 4; do echo "- {name: iid_s${i}_end, kind: checkpoint, path: $OUT/drive/dino_simsiam_iid45/checkpoints/s$((i+1))/final.pt, fewshot: false}" >> $f; done
  for i in 0 1 2; do echo "- {name: coco_s${i}_end, kind: checkpoint, path: $SH/offdomain/dino_simsiam/coco/checkpoints/s$i/final.pt, fewshot: false}" >> $f; done
  for i in 0 1 2 3 4; do echo "- {name: places365_s${i}_end, kind: checkpoint, path: $SH/offdomain/dino_simsiam/places365/checkpoints/s$i/final.pt, fewshot: false}" >> $f; done
  [ -f $OUT/ladder/dino_simsiam_controls/summary.json ] || run 0 python scripts/read_adaptation.py $GATE "$SCENES" \
    encoder=vit_b16_dino backbone=je "$TASKS" "$FT" fewshot.enabled=false fewshot.lr=1e-5 '~encoders' encoders_file=$f \
    hydra.run.dir=$OUT/ladder/dino_simsiam_controls
  # the same val pool, so the controls' reads join the arm's ladder
  cmp -s $OUT/ladder/dino_simsiam/pool.npz $OUT/ladder/dino_simsiam_controls/pool.npz || { echo "POOL MISMATCH"; return 1; }
  for n in $OUT/ladder/dino_simsiam_controls/{iid,coco,places365}_s*_end.npz; do ln -sfn ../dino_simsiam_controls/$(basename $n) $OUT/ladder/dino_simsiam/; done
}

analyze() {  # the analysis over every ladder present (host, no card): gates, breadth, curve, attribution, carry-forward
  uv run python scripts/analyze_adaptation.py mode=p125 root=$OUT n_iid_seeds=5 \
    'p125.panel=[weather,"large_cell:car,bus,truck:0.02","presence_cell:bus","presence_cell:pedestrian"]' \
    '+p125.attribution={mae_je:{coco:3,places365:5},dino_simsiam:{coco:3,places365:5}}' \
    "+p125.reference_floors={sup:$OUT/floors/sup}" hydra.run.dir=$OUT/analysis
}

# Part F1 (added 2026-11-10): the size-binned presence tasks, frozen rungs only (kNN, linear; no cls, no few-shot), on
# every arm's floor, walk checkpoints and IID ends -> outputs/p125/ladderF/<arm>. Same pool flags as GATE, so the query
# set is the ladders'. Bins by the largest box's frame share: under one 16 px patch at 224 px (1/196), 1-4 patches, 4+.
SUB=0.005102; LARGE=0.02
F1TASKS='tasks=['
for c in "traffic sign" pedestrian truck bus bicycle rider motorcycle; do
  F1TASKS+="\"presence_cell:$c\",\"size_cell:$c:0:$SUB\",\"size_cell:$c:$SUB:$LARGE\",\"size_cell:$c:$LARGE:inf\","
done
F1TASKS="${F1TASKS%,}]"

f1read() {  # arm card encoder flags...: one arm's F1 read (its existing encoder list, controls dropped, few-shot off)
  arm=$1; card=$2; shift 2
  f=$OUT/ladderF/encoders_$arm.yaml; mkdir -p $OUT/ladderF
  cat $OUT/ladder/encoders_$arm.yaml $([ $arm = dino_simsiam ] && echo $OUT/ladder/encoders_dino_simsiam_controls.yaml) \
    | grep -E "name: (floor|walk_s|iid_s)" | sed 's/fewshot: true/fewshot: false/' > $f
  [ -f $OUT/ladderF/$arm/summary.json ] || run $card python scripts/read_adaptation.py $GATE "$SCENES" "$@" "$F1TASKS" \
    cls_rungs=false fewshot.enabled=false 'fewshot.tasks=[]' '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladderF/$arm
  cmp -s <(uv run python -c "import numpy as n;print(n.load('$OUT/ladder/$arm/pool.npz')['query_index'].tolist())") \
    <(uv run python -c "import numpy as n;print(n.load('$OUT/ladderF/$arm/pool.npz')['query_index'].tolist())") \
    || { echo "QUERY MISMATCH $arm"; return 1; }
}

partFa() {  # card 0: DINO-init SimSiam, then DINO-init DINO
  f1read dino_simsiam 0 encoder=vit_b16_dino backbone=je
  f1read dino_dino 0 $DINO
}

partFb() {  # card 1: MAE-init JE, once card 0 has decoded the val pixel cache (both would write the same cache file)
  until [ -f $OUT/ladderF/dino_simsiam/floor.npz ]; do sleep 30; done
  f1read mae_je 1 encoder=vit_b16_mae backbone=je
}

partFc() {  # cards 0 / 1, after partFa / partFb: F1 on P1.2.1's two arms (MAE-init JE without protections; MAE's own
  # objective) -> outputs/p121/ladderF/<bb>. P1.2.5's pool flags, so the mean-pooled readout (P1.2.1 read `embed`):
  # internally consistent, not comparable to P1.2.1's own numbers. Same val query (checked).
  P121=outputs/p121
  for bc in je:0:partFa mae:1:partFb; do
    IFS=: read bb card after <<< "$bc"
    (
      until grep -q "P125_THIRD_${after}_DONE" $OUT/third_$after.log; do sleep 30; done
      f=$P121/ladderF/encoders_$bb.yaml; mkdir -p $P121/ladderF
      sed 's/fewshot: true/fewshot: false/' $P121/encoders_$bb.yaml > $f
      [ -f $P121/ladderF/$bb/summary.json ] || run $card python scripts/read_adaptation.py $GATE "$SCENES" encoder=vit_b16_mae \
        backbone=$bb "$F1TASKS" cls_rungs=false fewshot.enabled=false 'fewshot.tasks=[]' '~encoders' encoders_file=$f \
        hydra.run.dir=$P121/ladderF/$bb
      cmp -s <(uv run python -c "import numpy as n;print(n.load('$P121/ladder/$bb/pool.npz')['query_index'].tolist())") \
        <(uv run python -c "import numpy as n;print(n.load('$P121/ladderF/$bb/pool.npz')['query_index'].tolist())") \
        || echo "QUERY MISMATCH p121 $bb"
    ) &
  done
  wait
}

reads() {
  until grep -q P125_SECOND_DONE $OUT/second.log; do sleep 60; done
  for fl in dino:vit_b16_dino:1e-5 sup:vit_b16_pretrained:1e-4; do
    IFS=: read name enc lr <<< "$fl"
    [ -f $OUT/floors/$name/summary.json ] || run 0 python scripts/read_adaptation.py $GATE "$SCENES" encoder=$enc "$TASKS" "$FT" \
      fewshot.enabled=true fewshot.lr=$lr '~encoders' "+encoders=[{name:$name,kind:pretrained}]" hydra.run.dir=$OUT/floors/$name
  done
  f=$OUT/ladder/encoders_mae_je.yaml; mkdir -p $OUT/ladder
  echo "- {name: floor, kind: well, path: wells/p125/mae_je_heads.pt}" > $f
  walk_list $OUT/drive/mae_je_walk 1 5 >> $f
  for i in 0 1 2 3 4; do echo "- {name: iid_s${i}_end, kind: checkpoint, path: $OUT/drive/mae_je_iid/checkpoints/s$((i+1))/final.pt}" >> $f; done
  for i in 0 1 2; do echo "- {name: coco_s${i}_end, kind: checkpoint, path: $OUT/offdomain/drive/coco/checkpoints/s$i/final.pt}" >> $f; done
  for i in 0 1 2 3 4; do echo "- {name: places365_s${i}_end, kind: checkpoint, path: $OUT/offdomain/drive/places365/checkpoints/s$i/final.pt}" >> $f; done
  [ -f $OUT/ladder/mae_je/summary.json ] || run 0 python scripts/read_adaptation.py $GATE "$SCENES" encoder=vit_b16_mae backbone=je \
    "$TASKS" "$FT" fewshot.enabled=false fewshot.lr=3e-5 '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder/mae_je
  f=$OUT/ladder/encoders_dino_simsiam.yaml
  echo "- {name: floor, kind: well, path: wells/p125/dino_simsiam_heads.pt}" > $f
  walk_list $OUT/drive/dino_simsiam_walk 1 5 >> $f
  for i in 0 1 2; do echo "- {name: iid_s${i}_end, kind: checkpoint, path: $OUT/drive/dino_simsiam_iid/checkpoints/s$((i+1))/final.pt, fewshot: false}" >> $f; done
  [ -f $OUT/ladder/dino_simsiam/summary.json ] || run 0 python scripts/read_adaptation.py $GATE "$SCENES" encoder=vit_b16_dino \
    backbone=je "$TASKS" "$FT" fewshot.enabled=true fewshot.lr=1e-5 '~encoders' encoders_file=$f hydra.run.dir=$OUT/ladder/dino_simsiam
}

# One braced block ending in exit: bash parses it whole before running, so editing this file mid-run is safe.
{ "$1" > $OUT/third_$1.log 2>&1; echo "P125_THIRD_$1_DONE" >> $OUT/third_$1.log; exit; }
