#!/usr/bin/env bash

SCRIPT_DIR="$(dirname $0)"
CONFIG_DIR="$(dirname $0)/configs"
CONFIG_FILE="$CONFIG_DIR/Cityscapes-PanopticSegmentation/panoptic_uncertainty.yaml"

python $SCRIPT_DIR/train_net.py --config-file $CONFIG_FILE
