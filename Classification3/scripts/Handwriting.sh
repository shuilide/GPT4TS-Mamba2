for lr in 0.0001
do
for patch in 32
do
for stride in 8
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name Handwriting \
    --records_file Classification_records.xls \
    --data_dir ./datasets/Handwriting \
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
    --lr_step 30,60,90 \
    --lr_factor 0.5 \
    --gpu 5 \
    --seed 42

done
done
done