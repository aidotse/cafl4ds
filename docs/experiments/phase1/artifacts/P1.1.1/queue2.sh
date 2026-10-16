#!/bin/bash
# P1.1.1 follow-up queue (2026-10-16): fresh healthy seeds 10-14 per backbone (re-check of the revised
# false-alarm rule), and the warm collapse dial at a higher lr (1e-4) on the 54-step IID branch.
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
card1() {
  run 1 heldout2/mae_iid backbone=mae optim.lr=1e-4 stream.order=iid well=wells/p110_deep/mae_o0.0.pt 'seeds=[10,11,12,13,14]'
  for a in 0.3 1.0; do
    run 1 dial/je_iid_alpha${a}_lr1e-4 backbone=je optim.lr=1e-4 stream.order=iid well=wells/p110_deep/je_o0.0.pt \
      "resume.checkpoint='$P110/warm/je_o0.0_iid/checkpoints/s{seed}/step00065.pt'" resume.step=65 \
      '+ssl.collapse_alpha='$a 'seeds=[0]'
  done
}
card0() {
  while pgrep -f "outputs/p111/queue.sh" >/dev/null; do sleep 30; done   # wait for the overnight queue's last arm
  run 0 heldout2/je_iid backbone=je optim.lr=3e-5 stream.order=iid well=wells/p110_deep/je_o0.0.pt 'seeds=[10,11,12,13,14]'
  run 0 dial/je_iid_alpha0.0_lr1e-4 backbone=je optim.lr=1e-4 stream.order=iid well=wells/p110_deep/je_o0.0.pt \
    "resume.checkpoint='$P110/warm/je_o0.0_iid/checkpoints/s{seed}/step00065.pt'" resume.step=65 'seeds=[0]'
}
card0 & card1 & wait
echo "$(date +%F\ %T) queue2 done" >> $OUT/logs/progress.txt
