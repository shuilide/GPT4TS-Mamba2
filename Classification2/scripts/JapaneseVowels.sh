for lr in 0.0001
do
for patch in 4
do
for stride in 1
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name JapaneseVowels \
    --records_file Classification_records.xls \
    --data_dir ./datasets/JapaneseVowels \
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