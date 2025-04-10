GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main_test.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path scratch_training/OVGANet.pth \
        --options text_encoder_type="bert-base-uncased"
