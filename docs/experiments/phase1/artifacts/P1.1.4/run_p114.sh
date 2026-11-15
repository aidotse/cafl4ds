#!/bin/bash
# P1.1.4 — the reading foundation for P1.3's setup (design: P1.1.4.md). Stage 1, one mode per call:
#   card0 (card 0): DINO-init SimSiam's IID seeds 6-15 (protections, lr 9e-5), then the dose pilots of both JE arms
#          and of DINO-init DINO on their walks' step-846 states (DINO's damage hits student and teacher alike:
#          cafl4ds.branch.damage_method), then MAE-init JE's IID seeds 6-15 (protections, lr 3e-5), then the
#          provisional fourth arm, DINO-init DINO's IID seeds 4-15 (protections, lr 9e-5; no well: DINO's full
#          checkpoint resumes, its centre primed before step 0).
#   card1 (card 1): the MAE wells on P1.2.5's split (the 512-d x 8 decoder, and the 64-d x 2 one), the repaired MAE
#          walk x seeds 1-5 (P1.2.2's recipe: protections, lr 1e-4), the natural failure (P1.2.1's recipe: 64-d decoder,
#          no protections, lr 1e-4) walk x seeds 1-3, the repaired MAE's dose pilot, then its IID x seeds 1-15.
#   reads (both cards, after card0/card1), or `reads1 <je|dd|mae> <card>` one group at a time: the val ladder at every checkpoint of every Stage-1 drive — kNN, linear and
#          the 10- and 100-shot fine-tunes on scene and traffic light — on P1.2.5's val query frames (checked).
# Every drive is P1.2.5's (its diet, split, health vector every 25 steps, Part B's checkpoints) and lands on the
# project share, linked from outputs/p114. Run from the main checkout, which holds wells/ and outputs/.
# Stage 2, `stage2 <card>`: the damage branches of the arms on that card (card 0: DINO-init SimSiam and DINO-init
#   DINO; card 1: MAE-init JE and MAE), then their val ladder. Each branch resumes its walk's step-846 state (seeds
#   1-3), damages the encoder (the dose's draw varies with the drive seed, as in the pilots), saves that start state,
#   and drives the rest of the walk, saving step 1,269 and the end. Per arm: the dose-0 twin, four noise doses read
#   off the 3-seed pilots (2-15 linear-canary points), and one re-init (the specificity control where the pilot
#   passes it, else one more dose; P1.1.4.md). Then `features <card>`: the block features of every checkpoint the
#   reader screen reads (branches, walks, IID drives, natural failures), on the same cards' arms.
set -u
cd /home/mauricio/projects/cafl4ds
SCENES='data.scenes=["city street",highway,residential]'
BDD=/mnt/pr_2018_scaleout_workdir/cafl4ds/data/bdd100k; OUT=outputs/p114
SHARE=/mnt/pr_2018_scaleout_workdir/cafl4ds; SH=$SHARE/outputs/p114
mkdir -p $SH; [ -e $OUT ] || ln -s $SH $OUT
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /home/mauricio/.cache/torch -m $SHARE:$SHARE \
  gaudi-env-cafl4ds:latest $card "$@"; }
DIET="data=bdd data.split=train data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache bdd_root=$BDD \
 device=hpu img_size=224 stream.support_per_canary=100 stream.query_per_canary=100 stream.stratify_canary=true \
 stream.well_fraction=0.05 stream.select_support_per_canary=450 stream.select_query_per_canary=450"
PROTECT="gate_arms=false eval_every=25 param_groups.layer_decay=0.75 param_groups.no_decay=true study=P1.1.4"
NOPROT="gate_arms=false eval_every=25 study=P1.1.4"
CKPTS=checkpoint_legs=steps; STEPS='checkpoint_steps=[100,0.125,0.25,0.5,0.75]'  # Part B's checkpoints
SIM="encoder=vit_b16_dino backbone=je"; DINO="encoder=vit_b16_dino backbone=dino"; MAEJE="encoder=vit_b16_mae backbone=je"; MAE="encoder=vit_b16_mae backbone=mae"
DEC='ssl.decoder_dim=512 ssl.decoder_depth=8 ssl.decoder_heads=16 ssl.decoder_mlp_ratio=4.0'
WS=wells/p125/dino_simsiam_heads.pt; WM=wells/p125/mae_je_heads.pt
WR=wells/p114/mae_dec512_heads.pt; WF=wells/p114/mae_dec64_heads.pt
DOSES="doses=[reinit:4,reinit:12,noise:12:0.1,noise:12:0.2,noise:12:0.3,noise:12:0.4,noise:12:0.5,noise:12:0.75,noise:12:1.0]"

link() {  # a run dir on the share, linked from outputs/p114
  mkdir -p $SH/$1 $(dirname $OUT/$1); [ -e $OUT/$1 ] || ln -s $SH/$1 $OUT/$1
}

drive() {  # card dir last-seed recipe-flags...: one drive set, skipped once its last seed's end exists
  card=$1; dir=$2; last=$3; shift 3
  link $dir
  [ -f $OUT/$dir/checkpoints/$last/final.pt ] || run $card python scripts/run_deploy_harness.py "$@" hydra.run.dir=$SH/$dir
}

pilot() {  # card name checkpoint-pattern well recipe-flags...: the probe-only dose ladder on a step-846 state
  card=$1; name=$2; ck=$3; well=$4; shift 4
  link pilot/$name
  [ -f $OUT/pilot/$name/dose_pilot.json ] || run $card python scripts/dose_pilot.py $DIET "$SCENES" "$@" well=$well \
    "resume.checkpoint='$ck'" seed=1 "$DOSES" hydra.run.dir=$SH/pilot/$name
}

# The refined pilot: three seeds' step-846 states, a finer noise grid where the first (seed-1) pilot left gaps; doses are
# chosen on the seed mean, never on one seed.
FINE="doses=[reinit:2,reinit:4,noise:12:0.2,noise:12:0.3,noise:12:0.4,noise:12:0.5,noise:12:0.55,noise:12:0.6,noise:12:0.65,noise:12:0.7,noise:12:0.75,noise:12:0.8,noise:12:0.875,noise:12:1.0]"
pilot3() {  # card name checkpoint-pattern well recipe-flags...: the refined pilot on seeds 1-3
  card=$1; name=$2; ck=$3; well=$4; shift 4
  for s in 1 2 3; do
    link pilot_fine/${name}_s$s
    [ -f $OUT/pilot_fine/${name}_s$s/dose_pilot.json ] || run $card python scripts/dose_pilot.py $DIET "$SCENES" "$@" \
      well=$well "resume.checkpoint='$ck'" seed=$s "$FINE" hydra.run.dir=$SH/pilot_fine/${name}_s$s
  done
}

pilots_fine() {  # card
  pilot3 $1 dino_simsiam "outputs/p125/partG/drive/dino_simsiam_p9e-5_walk/checkpoints/s{seed}/step00846.pt" $WS $SIM
  pilot3 $1 mae_je "outputs/p125/drive/mae_je_walk/checkpoints/s{seed}/step00846.pt" $WM $MAEJE
  pilot3 $1 dino_dino "outputs/p125/drive/dino_dino_walk/checkpoints/s{seed}/step00846.pt" null $DINO
  pilot3 $1 mae "$OUT/drive/mae_walk/checkpoints/s{seed}/step00846.pt" $WR $MAE $DEC
}

well() {  # card out decoder-flags...: a MAE well, heads settled on the frozen encoder (P1.2.1 / P1.2.2's settings)
  card=$1; out=$2; shift 2
  mkdir -p wells/p114
  [ -f $out ] || run $card python scripts/warm_well.py $DIET "$SCENES" $MAE "$@" optim.lr=3e-4 warmup.freeze_encoder=true \
    warmup.stop=loss warmup.window=200 warmup.tol=0.002 warmup.min_steps=500 warmup.max_steps=3000 warmup.max_passes=100 \
    warmup.select_every=500 well_out=$out hydra.run.dir=$SH/wells/$(basename $out .pt)
}

GATE="data=bdd bdd_root=$BDD data.labels_file=$BDD/geometry/bdd100k_labels_images_val_box2d.json data.boxes=true \
 data.drivable_dir=$BDD/geometry/drivable/labels/val data.min_canary_count=50 data.pixel_dtype=uint8 \
 data.cache_dir=/workspace/outputs/cache device=hpu img_size=224 pool.split=val pool.fit_fraction=0.4 threads=8 readout=mean"
GATETASKS='tasks=[scene,"presence_cell:traffic light"]'; FT='fewshot.tasks=[scene,"presence_cell:traffic light"]'

ckpt_list() {  # drive-dir name-prefix seeds...: one ladder entry per saved checkpoint, the fine-tunes on
  dir=$1; prefix=$2; shift 2
  for s in "$@"; do
    d=$dir/checkpoints/s$s
    for c in $(python3 -c "import json;print(' '.join(e['file'][:-3] for e in json.load(open('$d/index.json'))))"); do
      t=${c/step/t}; [ "$c" = final ] && t=end
      echo "- {name: ${prefix}_s${s}_$t, kind: checkpoint, path: $d/$c.pt, fewshot: true}"
    done
  done
}

ladder() {  # card name floor ft-lr encoder-flags... -- list lines on stdin: the val ladder of one arm's drives;
  # floor is the arm's settled-head well, or `pretrained` (DINO-init DINO: the released backbone, as P1.2.5 read it)
  card=$1; name=$2; floor=$3; lr=$4; shift 4
  link ladder/$name
  [ -f $OUT/ladder/$name/summary.json ] && return
  # claim the ladder, so a group queued on both cards runs each ladder on one card only (a failed read leaves the
  # claim: remove $SH/ladder/$name.claimed to retry)
  mkdir $SH/ladder/$name.claimed 2>/dev/null || { echo "skip $name: claimed by another card"; return; }
  f=$SH/ladder/encoders_$name.yaml
  if [ "$floor" = pretrained ]; then fl="- {name: floor, kind: pretrained, fewshot: true}"
  else fl="- {name: floor, kind: well, path: $floor, fewshot: true}"; fi
  { echo "$fl"; cat; } > $f
  [ -f $OUT/ladder/$name/summary.json ] || run $card python scripts/read_adaptation.py $GATE cls_rungs=false "$SCENES" \
    "$@" "$GATETASKS" "$FT" fewshot.enabled=true fewshot.lr=$lr '~encoders' encoders_file=$f hydra.run.dir=$SH/ladder/$name
  cmp -s <(uv run python -c "import numpy as n;print(n.load('outputs/p125/ladder/dino_simsiam/pool.npz')['query_index'].tolist())") \
    <(uv run python -c "import numpy as n;print(n.load('$OUT/ladder/$name/pool.npz')['query_index'].tolist())") \
    || echo "QUERY MISMATCH $name"
}

reads_je() {  # card: the JE arms' new IID drives (fine-tunes at the floor's rate: DINO 1e-5, MAE-init 3e-5)
  ckpt_list $OUT/drive/dino_simsiam_iid_6to15 iid $(seq 6 15) | ladder $1 dino_simsiam_iid $WS 1e-5 $SIM
  ckpt_list $OUT/drive/mae_je_iid_6to15 iid $(seq 6 15) | ladder $1 mae_je_iid $WM 3e-5 $MAEJE
}

reads_dd() {  # card: DINO-init DINO's new IID drives
  ckpt_list $OUT/drive/dino_dino_iid_4to15 iid $(seq 4 15) | ladder $1 dino_dino_iid pretrained 1e-5 $DINO
}

reads_mae() {  # card: the MAE arms (P1.2.2's few-shot rate, 3e-5)
  { ckpt_list $OUT/drive/mae_walk walk 1 2 3 4 5; ckpt_list $OUT/drive/mae_iid iid $(seq 1 15); } \
    | ladder $1 mae $WR 3e-5 $MAE $DEC
  ckpt_list $OUT/drive/mae_fail_walk walk 1 2 3 | ladder $1 mae_fail $WF 3e-5 $MAE
}

# Stage 2 — the damage branches. branches <card> <arm> <walk-ckpt-pattern> <well|null> <reinit-k> <doses> -- recipe...
branches() {
  card=$1; arm=$2; ck=$3; well=$4; rk=$5; doses=$6; shift 6
  w=""; [ "$well" = null ] || w="well=$well"
  kinds="dose0:none:0:0 reinit$rk:reinit:$rk:0"
  for d in $doses; do kinds="$kinds noise${d/./p}:noise:12:$d"; done
  for kind in $kinds; do
    IFS=: read -r name mode k scale <<< "$kind"
    drive $card branch/${arm}_$name s3 $DIET "$SCENES" "$@" $w "resume.checkpoint='$ck'" resume.step=846 \
      resume.save_start=true damage.mode=$mode damage.k=$k damage.scale=$scale $CKPTS "$STEPS" 'seeds=[1,2,3]'
  done
}

branch_reads() {  # card arm floor ft-lr encoder-flags...: the val ladder at each branch's start, step 1,269 and end
  card=$1; arm=$2; floor=$3; lr=$4; shift 4
  for d in $(br $arm | tr , ' '); do ckpt_list $d $(basename $d | sed "s/^${arm}_//") 1 2 3; done \
    | ladder $card branch_$arm $floor $lr "$@"
}

# An arm's branch dirs, comma-separated: its own kinds only (a bare `mae_*` would also match `mae_je_*`).
br() { ls -d $OUT/branch/${1}_dose0 $OUT/branch/${1}_reinit[0-9]* $OUT/branch/${1}_noise0p* 2>/dev/null | paste -sd,; }

branch_reads_mae() {  # card: the MAE arm's branch ladder alone (the first launcher's pattern caught MAE-init JE's dirs)
  branch_reads $1 mae $WR 3e-5 $MAE $DEC
}

# The walks' step-846 states and recipes (P1.2.5's walks for the JE starts, P1.1.4's for MAE); doses in SD multiples.
B_SIM="outputs/p125/partG/drive/dino_simsiam_p9e-5_walk/checkpoints/s{seed}/step00846.pt"
B_MJE="outputs/p125/drive/mae_je_walk/checkpoints/s{seed}/step00846.pt"
B_DD="outputs/p125/drive/dino_dino_walk/checkpoints/s{seed}/step00846.pt"
B_MAE="$OUT/drive/mae_walk/checkpoints/s{seed}/step00846.pt"

stage2_card0() {
  branches 0 dino_simsiam $B_SIM $WS 4 "0.5 0.65 0.75 0.875" $SIM $PROTECT optim.lr=9e-5 stream.order=walk
  branches 0 dino_dino $B_DD null 4 "0.3 0.5 0.65 0.75" $DINO $PROTECT optim.lr=9e-5 stream.order=walk
  branch_reads 0 dino_simsiam $WS 1e-5 $SIM
  branch_reads 0 dino_dino pretrained 1e-5 $DINO
}

stage2_card1() {
  branches 1 mae_je $B_MJE $WM 2 "0.3 0.4 0.5 0.65" $MAEJE $PROTECT optim.lr=3e-5 stream.order=walk
  branches 1 mae $B_MAE $WR 2 "0.4 0.5 0.65 0.875" $MAE $DEC $PROTECT optim.lr=1e-4 stream.order=walk
  branch_reads 1 mae_je $WM 3e-5 $MAEJE
  branch_reads 1 mae $WR 3e-5 $MAE $DEC
}

# The block features (P1.1.2's per-block reads) of every checkpoint an arm's reader screen reads: its branches, its
# walks, its IID drives with all six checkpoints, and its natural failure. One extraction per arm (one method); with no
# well (DINO-init DINO) the anchor is the method as built. features1 <card> <arm> <well|null> <dirs> -- recipe...
features1() {
  card=$1; arm=$2; well=$3; dirs=$4; shift 4
  link features/$arm
  [ -f $OUT/features/$arm/index.json ] && return
  run $card python scripts/extract_block_features.py $DIET "$SCENES" "$@" well=$well "lens.drives=[$dirs]" \
    hydra.run.dir=$SH/features/$arm
}

features_card0() {
  features1 0 dino_simsiam $WS "$(br dino_simsiam),outputs/p125/partG/drive/dino_simsiam_p9e-5_walk,$OUT/drive/dino_simsiam_iid_6to15,outputs/p125/partG/drive/dino_simsiam_np_walk" $SIM
  features1 0 dino_dino null "$(br dino_dino),outputs/p125/drive/dino_dino_walk,$OUT/drive/dino_dino_iid_4to15" $DINO
}

features_card1() {
  features1 1 mae_je $WM "$(br mae_je),outputs/p125/drive/mae_je_walk,$OUT/drive/mae_je_iid_6to15,outputs/p125/partG/drive/mae_je_np_walk" $MAEJE
  features1 1 mae $WR "$(br mae),$OUT/drive/mae_walk,$OUT/drive/mae_iid" $MAE $DEC
  features1 1 mae_fail $WF "$OUT/drive/mae_fail_walk" $MAE  # the 64-d decoder: its own method and anchor
}

card0() {
  drive 0 drive/dino_simsiam_iid_6to15 s15 $DIET "$SCENES" $SIM $PROTECT optim.lr=9e-5 stream.order=iid well=$WS $CKPTS "$STEPS" \
    'seeds=[6,7,8,9,10,11,12,13,14,15]'
  pilot 0 dino_simsiam "outputs/p125/partG/drive/dino_simsiam_p9e-5_walk/checkpoints/s{seed}/step00846.pt" $WS $SIM
  pilot 0 mae_je "outputs/p125/drive/mae_je_walk/checkpoints/s{seed}/step00846.pt" $WM $MAEJE
  pilot 0 dino_dino "outputs/p125/drive/dino_dino_walk/checkpoints/s{seed}/step00846.pt" null $DINO
  drive 0 drive/mae_je_iid_6to15 s15 $DIET "$SCENES" $MAEJE $PROTECT optim.lr=3e-5 stream.order=iid well=$WM $CKPTS "$STEPS" \
    'seeds=[6,7,8,9,10,11,12,13,14,15]'
  drive 0 drive/dino_dino_iid_4to15 s15 $DIET "$SCENES" $DINO $PROTECT optim.lr=9e-5 stream.order=iid $CKPTS "$STEPS" \
    'seeds=[4,5,6,7,8,9,10,11,12,13,14,15]'
}

card1() {
  well 1 $WR $DEC
  well 1 $WF
  drive 1 drive/mae_walk s5 $DIET "$SCENES" $MAE $DEC $PROTECT optim.lr=1e-4 stream.order=walk well=$WR $CKPTS "$STEPS" \
    'seeds=[1,2,3,4,5]'
  drive 1 drive/mae_fail_walk s3 $DIET "$SCENES" $MAE $NOPROT optim.lr=1e-4 stream.order=walk well=$WF $CKPTS "$STEPS" \
    'seeds=[1,2,3]'
  pilot 1 mae "$OUT/drive/mae_walk/checkpoints/s{seed}/step00846.pt" $WR $MAE $DEC
  drive 1 drive/mae_iid s15 $DIET "$SCENES" $MAE $DEC $PROTECT optim.lr=1e-4 stream.order=iid well=$WR $CKPTS "$STEPS" \
    'seeds=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]'
}

case "${1:-}" in
  card0) card0 > $SH/card0.log 2>&1 ;;
  card1) card1 > $SH/card1.log 2>&1 ;;
  reads) { reads_je 0; reads_dd 0; } > $SH/reads0.log 2>&1 & reads_mae 1 > $SH/reads1.log 2>&1 & wait ;;
  # one ladder group on a chosen card (a card freed early takes the drives already done): reads1 <je|dd|mae> <card>
  reads1) reads_$2 $3 > $SH/reads_$2.log 2>&1 ;;
  pilots_fine) pilots_fine $2 > $SH/pilots_fine.log 2>&1 ;;
  stage2) stage2_card$2 > $SH/stage2_card$2.log 2>&1 ;;
  features) features_card$2 > $SH/features_card$2.log 2>&1 ;;
  branch_reads_mae) branch_reads_mae $2 > $SH/branch_reads_mae.log 2>&1 ;;
  *) echo "usage: $0 card0|card1|reads|reads1 <je|dd|mae> <card>|pilots_fine <card>|stage2 <card>|features <card>|branch_reads_mae <card>"; exit 1 ;;
esac
echo "P114_${1}_DONE"
