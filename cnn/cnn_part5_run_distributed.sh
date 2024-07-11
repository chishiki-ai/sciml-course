#!/bin/bash
LOCAL_RANK=$PMI_RANK

NODEFILE="$WORK/hostfile"
scontrol show hostnames > $NODEFILE
if [[ -s "$NODEFILE" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi


PRELOAD="/opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv --bind /run/user:/run/user /work2/10000/zw427/cnn_course_latest.sif "
CMD="python3 -m torch.distributed.run --nproc_per_node 3 --nnodes $NNODES --node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK --master_port=1234 cnn_part5_torch_train_distributed.py $@"

FULL_CMD="$PRELOAD $CMD"
echo "Training command: $FULL_CMD"

eval $FULL_CMD