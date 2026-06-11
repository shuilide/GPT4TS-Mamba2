for lr in 0.0001
do
for patch in 32
do
for stride in 16
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name Heartbeat \
    --records_file Classification_records.xls \
    --data_dir ./datasets/Heartbeat \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 100 \
    --lr $lr \
    --patch_size $patch \
    --stride $stride \
    --optimizer AdamW \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --gpu 5 \
    --seed 42

done
done
done
