#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=Panoptic-DeepLab-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8             # CPU cores/threads
#SBATCH --gres=gpu:t4:2                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --mail-type=ALL
#SBATCH --array=1-2%1   # 4 is the number of jobs in the chain

module load singularity/3.8

SING_IMG=detectron2.sif

PROJ_DIR=$PWD
DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/cityscapes

TMP_DATA_DIR=$SLURM_TMPDIR/data/cityscapes

tar -zxf DATA_DIR/cityscapes.tar.gz -C TMP_DATA_DIR

BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
singularity exec
--env-file $PROJ_DIR/tools/envfile
--bind $PROJ_DIR:/Pan-DL/code
--bind $TMP_DATA_DIR:/Pan-DL/datasets/cityscapes
$SING_IMG
"

PAN_DL_DIR=/Pan-DL/code/projects/Panoptic-DeepLab

TRAIN_CMD="$BASE_CMD
python $PAN_DL_DIR/train_net.py
--config-file $PAN_DL_DIR/configs/Cityscapes-PanopticSegmentation/panoptic_uncertainty.yaml
"

TEST_CMD="$BASE_CMD
bash
"
eval $TEST_CMD

#eval $TRAIN_CMD
