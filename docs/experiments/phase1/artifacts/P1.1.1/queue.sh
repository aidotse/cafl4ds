#!/bin/bash
# P1.1.1 overnight queue — one worker per card (card 0 JE, card 1 MAE). Every arm resumes or drives
# from the P1.1.0 deep overlap-0 wells / leg checkpoints, on the P1.1.0 deep diet.
BDD=/mnt/pr_2018_scaleout_workdir/bdd100k; OUT=outputs/p111; P110=outputs/p110_deep
DIET="data=bdd data.split=val data.min_canary_count=50 bdd_root=$BDD device=hpu img_size=224 encoder=vit_b16_mae \
 stream.support_per_canary=45 stream.query_per_canary=45 stream.stratify_canary=true stream.well_fraction=0.5 \
 stream.select_support_per_canary=150 stream.select_query_per_canary=150 stream.well_overlap=0.0 \
 gate_arms=false eval_every=1 study=P1.1.1"
START=65   # the night·clear leg opens here; ~54 steps of drive remain after the injection
run() { card=$1; name=$2; shift 2
  if [ -f $OUT/$name/corpus/manifest.json ]; then echo "$(date +%F\ %T) skip $name (done)" >> $OUT/logs/progress.txt; return; fi
  echo "$(date +%F\ %T) start $name (card $card)" >> $OUT/logs/progress.txt
  ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card \
    python scripts/run_deploy_harness.py $DIET "$@" hydra.run.dir=$OUT/$name > $OUT/logs/$(echo $name | tr / _).log 2>&1
  echo "$(date +%F\ %T) end $name rc=$?" >> $OUT/logs/progress.txt
}
branch() { card=$1; bb=$2; lr=$3; order=$4; dose=$5; shift 5   # remaining args: the damage / dial overrides
  run $card branch/${bb}_${order}_$dose backbone=$bb optim.lr=$lr stream.order=$order well=wells/p110_deep/${bb}_o0.0.pt \
    "resume.checkpoint='$P110/warm/${bb}_o0.0_$order/checkpoints/s{seed}/step000$START.pt'" resume.step=$START \
    checkpoint_legs=final 'seeds=[0,1,2]' "$@"
}
worker() { card=$1; bb=$2; lr=$3
  if [ $bb = je ]; then   # warm collapse-dial pilot (1 seed): does the dial bite within 54 steps from the well?
    for a in 0.1 0.3 1.0; do
      run $card dial/je_iid_alpha$a backbone=je optim.lr=$lr stream.order=iid well=wells/p110_deep/je_o0.0.pt \
        "resume.checkpoint='$P110/warm/je_o0.0_iid/checkpoints/s{seed}/step000$START.pt'" resume.step=$START \
        '+ssl.collapse_alpha='$a 'seeds=[0]'
    done
  fi
  run $card heldout/${bb}_iid backbone=$bb optim.lr=$lr stream.order=iid well=wells/p110_deep/${bb}_o0.0.pt 'seeds=[5,6,7,8,9]'
  for order in iid walk; do
    branch $card $bb $lr $order none
    branch $card $bb $lr $order noise0.5 damage.mode=noise damage.k=12 damage.scale=0.5
    branch $card $bb $lr $order noise1.0 damage.mode=noise damage.k=12 damage.scale=1.0
    branch $card $bb $lr $order reinit12 damage.mode=reinit damage.k=12
    branch $card $bb $lr $order noise0.75 damage.mode=noise damage.k=12 damage.scale=0.75
    branch $card $bb $lr $order noise0.3 damage.mode=noise damage.k=12 damage.scale=0.3
    branch $card $bb $lr $order reinit4 damage.mode=reinit damage.k=4
  done
  for order in walk iid; do   # recovery: 60 stationary well-pool steps from the end of each P1.1.0 drive
    run $card recover/${bb}_$order backbone=$bb optim.lr=$lr well=wells/p110_deep/${bb}_o0.0.pt \
      "resume.checkpoint='$P110/warm/${bb}_o0.0_$order/checkpoints/s{seed}/final.pt'" recovery_steps=60 'seeds=[0,1,2]'
  done
}
worker 0 je 3e-5 & worker 1 mae 1e-4 & wait
echo "$(date +%F\ %T) queue done" >> $OUT/logs/progress.txt
