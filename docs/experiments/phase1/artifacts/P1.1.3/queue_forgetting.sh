#!/bin/bash
# P1.1.3 item 5 — re-run P0.3.9's seed-stable natural MAE forgetting (deep decoder-first well, epochs_b=40)
# with save_states=true, so each arm's post-A and post-B method can be read per block. One worker per card.
# Each writes outputs/p113/forgetting/epochsb40_seed<s>/ (comparison.json + states/, ~1.3 GB per seed).
ARGS="device=hpu training_mode=ssl pretrained_encoder=true skip_phase_a=false decoder_warmup_epochs=30 \
 encoder=vit_b16_mae encoder.state_dict_path=/mnt/cifar100/mae_pretrain_vit_base.pth \
 ssl=mae ssl.decoder_dim=512 ssl.decoder_depth=8 ssl.decoder_heads=16 \
 data=imagenette data_b=cifar100 max_per_class=260 \
 task_a_classes=[0,1,2,3,4,5,6,7,8,9] task_b_classes=[0,1,2,3,4,5,6,7,8,9] \
 support_per_class=60 query_per_class=60 \
 epochs_a=60 epochs_b=40 batch_size=64 probe=linear recon_masks=8 optim.lr=3e-4 save_states=true"
worker() {
  card=$1; shift
  for s in "$@"; do
    dir=outputs/p113/forgetting/epochsb40_seed$s
    ./scripts/run_gaudi_dev.sh -m /mnt gaudi-env-cafl4ds:latest $card \
      python scripts/positive_control_forgetting.py $ARGS seed=$s hydra.run.dir=$dir > $dir.log 2>&1
  done
}
mkdir -p outputs/p113/forgetting
worker 0 0 2 4 & worker 1 1 3 & wait
