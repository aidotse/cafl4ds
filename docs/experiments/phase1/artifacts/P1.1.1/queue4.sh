#!/bin/bash
# P1.1.1 warm collapse dial, dose sweep (2026-10-16): the 54-step branch at lr 1e-4 bites at alpha 1.0 (seed-0 pilot),
# so alpha in {0, 0.6, 1.0} x seeds 0-2 on the IID branch, and alpha in {0, 1.0} on the walk branch. Starts each
# card once queue3's worker on that card is done.
BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p111; P110=outputs/p110_deep
DIET="data=bdd data.split=val data.min_canary_count=50 bdd_root=$BDD device=hpu img_size=224 encoder=vit_b16_mae \
 stream.support_per_canary=45 stream.query_per_canary=45 stream.stratify_canary=true stream.well_fraction=0.5 \
 stream.select_support_per_canary=150 stream.select_query_per_canary=150 stream.well_overlap=0.0 \
 gate_arms=false eval_every=1 study=P1.1.1"
run() { card=$1; name=$2; shift 2
  if [ -f $OUT/$name/corpus/manifest.json ]; then echo "$(date +%F\ %T) skip $name (done)" >> $OUT/logs/progress.txt; return; fi
  echo "$(date +%F\ %T) start $name (card $card)" >> $OUT/logs/progress.txt
  ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card \
    python scripts/run_deploy_harness.py $DIET "$@" hydra.run.dir=$OUT/$name > $OUT/logs/$(echo $name | tr / _).log 2>&1
  echo "$(date +%F\ %T) end $name rc=$?" >> $OUT/logs/progress.txt
}
arm() { card=$1; order=$2; a=$3
  run $card dial/je_${order}_alpha${a}_lr1e-4_s012 backbone=je optim.lr=1e-4 stream.order=$order \
    well=wells/p110_deep/je_o0.0.pt "resume.checkpoint='$P110/warm/je_o0.0_${order}/checkpoints/s{seed}/step00065.pt'" \
    resume.step=65 "+ssl.collapse_alpha=$a" 'seeds=[0,1,2]'
}
# $1 / $2: the PIDs of queue3's card-0 / card-1 workers (each card starts once its worker exits).
waitfor() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
card0() { waitfor "$1"; arm 0 iid 0.0; arm 0 iid 1.0; arm 0 iid 0.6; }
card1() { waitfor "$1"; arm 1 walk 0.0; arm 1 walk 1.0; }
card0 "$1" & card1 "$2" & wait
echo "$(date +%F\ %T) queue4 done" >> $OUT/logs/progress.txt
