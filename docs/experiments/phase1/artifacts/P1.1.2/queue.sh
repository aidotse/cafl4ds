#!/bin/bash
# P1.1.2 Stage 1 — per-block feature extraction from the saved P1.1.0 / P1.1.1 checkpoints.
# One worker per card (JE on 0, MAE on 1). Each writes outputs/p112/features_<bb>/; reruns skip existing files.
BDD=/mnt/pr_2018_scaleout_workdir/bdd100k
DIET="data=bdd data.split=val data.min_canary_count=50 bdd_root=$BDD device=hpu img_size=224 encoder=vit_b16_mae \
 stream.support_per_canary=45 stream.query_per_canary=45 stream.stratify_canary=true stream.well_fraction=0.5 \
 stream.select_support_per_canary=150 stream.select_query_per_canary=150 stream.well_overlap=0.0 \
 gate_arms=false eval_every=1 study=P1.1.2"
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $BDD -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card "$@"; }
extract() {
  card=$1; bb=$2
  arms=$(ls -d outputs/p111/branch/${bb}_* | paste -sd, -)
  run $card python scripts/extract_block_features.py $DIET backbone=$bb well=wells/p110_deep/${bb}_o0.0.pt \
    "lens.drives=[outputs/p110_deep/warm/${bb}_o0.0_iid,outputs/p110_deep/warm/${bb}_o0.0_walk]" \
    "lens.branches=[$arms]" hydra.run.dir=outputs/p112/features_${bb} > outputs/p112/features_${bb}.log 2>&1
}
mkdir -p outputs/p112
extract 0 je & extract 1 mae & wait
