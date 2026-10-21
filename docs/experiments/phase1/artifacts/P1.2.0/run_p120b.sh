#!/bin/bash
# P1.2.0, second batch (2026-10-21): the extended screen under squash (card 0) and centre crop (card 1), then the
# held-out ladder re-read on the three canary scenes (the first read kept parking lot: outputs/p120/v1_4scene/).
set -u
cd /home/mauricio/projects/cafl4ds
TRAIN=/home/mauricio/data/bdd100k; OUT=outputs/p120
run() { card=$1; shift; ./scripts/run_gaudi_dev.sh -m $TRAIN -m /mnt/cifar100 gaudi-env-cafl4ds:latest $card "$@"; }
SCENES='data.scenes=["city street",highway,residential]'
POOL="data=bdd bdd_root=$TRAIN data.min_canary_count=50 data.pixel_dtype=uint8 data.cache_dir=/workspace/outputs/cache \
 device=hpu img_size=224 encoder=vit_b16_mae pool.split=train pool.max_images=14000 threads=8"
CANDS='screen.candidates=[weather,"presence_cell:car","presence_cell:truck","presence_cell:pedestrian","presence_cell:traffic light"]'
TASKS='tasks=[scene,"presence:traffic light","presence:pedestrian","presence:truck"]'

screen() {  # card, resize mode
  run $1 python scripts/read_adaptation.py $POOL "$SCENES" data.resize_mode=$2 mode=screen fewshot.lr=3e-5 \
    fewshot.select_k=100 fewshot.draws=3 screen.reference=scene "$CANDS" screen.max_overlap=0.60 \
    screen.min_headroom=0.10 hydra.run.dir=$OUT/screen2/$2
}
ladder() {  # card, backbone (encoders_<bb>.yaml from the first batch)
  run $1 python scripts/read_adaptation.py $POOL "$SCENES" "$TASKS" backbone=$2 fewshot.enabled=true \
    'fewshot.tasks=[scene]' fewshot.lr=3e-5 '~encoders' encoders_file=$OUT/encoders_$2.yaml hydra.run.dir=$OUT/ladder/$2
}
( screen 0 squash; ladder 0 je ) > $OUT/batch2_card0.log 2>&1 &
( screen 1 center_crop; ladder 1 mae ) > $OUT/batch2_card1.log 2>&1 &
wait
echo P120B_DONE
