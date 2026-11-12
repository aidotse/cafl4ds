#!/bin/bash
# P1.1.1 collapse-dial fallback (2026-10-16): the 54-step branch at lr 1e-4 did not bite at alpha 0.3, so
# (a) full 119-step IID drives from the well at lr 1e-4, alpha 1.0 and its alpha 0 twin; (b) P0.2.5's other
# knob — soft stop-gradient off (stopgrad_beta 0) — on the 54-step branch at lr 1e-4. Seed 0 each.
BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p111; P110=outputs/p110_deep
DIET="data=bdd data.split=val data.min_canary_count=50 bdd_root=$BDD device=hpu img_size=224 encoder=vit_b16_mae \
 stream.support_per_canary=45 stream.query_per_canary=45 stream.stratify_canary=true stream.well_fraction=0.5 \
 stream.select_support_per_canary=150 stream.select_query_per_canary=150 stream.well_overlap=0.0 \
 gate_arms=false eval_every=1 study=P1.1.1"
JE="backbone=je optim.lr=1e-4 stream.order=iid well=wells/p110_deep/je_o0.0.pt seeds=[0]"
BRANCH="resume.checkpoint='$P110/warm/je_o0.0_iid/checkpoints/s{seed}/step00065.pt' resume.step=65"
run() { card=$1; name=$2; shift 2
  if [ -f $OUT/$name/corpus/manifest.json ]; then echo "$(date +%F\ %T) skip $name (done)" >> $OUT/logs/progress.txt; return; fi
  echo "$(date +%F\ %T) start $name (card $card)" >> $OUT/logs/progress.txt
  ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card \
    python scripts/run_deploy_harness.py $DIET "$@" hydra.run.dir=$OUT/$name > $OUT/logs/$(echo $name | tr / _).log 2>&1
  echo "$(date +%F\ %T) end $name rc=$?" >> $OUT/logs/progress.txt
}
card0() {
  run 0 dial/je_iid_full_alpha1.0_lr1e-4 $JE '+ssl.collapse_alpha=1.0'
  run 0 dial/je_iid_full_alpha0.0_lr1e-4 $JE
}
card1() {
  while pgrep -f "outputs/p111/queue2.sh" >/dev/null; do sleep 30; done
  run 1 dial/je_iid_beta0.0_lr1e-4 $JE $BRANCH '+ssl.stopgrad_beta=0.0'
}
case "$1" in card1) card1 ;; *) card0 & card1 & wait ;; esac
echo "$(date +%F\ %T) queue3 done" >> $OUT/logs/progress.txt
