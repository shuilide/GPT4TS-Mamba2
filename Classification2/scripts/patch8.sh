#python src/main.py \
#    --output_dir experiments \
#    --comment "classification from Scratch" \
#    --name AtrialFibrillation \
#    --records_file Classification_records.xls \
#    --data_dir ./datasets/AtrialFibrillation \
#    --data_class tsra \
#    --pattern TRAIN \
#    --val_pattern TEST \
#    --epochs 100 \
#    --lr 0.0005 \
#    --patch_size 8 \
#    --stride 4 \
#    --optimizer AdamW \
#    --d_model 768 \
#    --pos_encoding learnable \
#    --task classification \
#    --key_metric accuracy \
#    --gpu 5 \
#    --seed 42 \
#    --gpt_layers 6 \
#    --num_mamba_layers 2
#
#python src/main.py \
#    --output_dir experiments \
#    --comment "classification from Scratch" \
#    --name BasicMotions \
#    --records_file Classification_records.xls \
#    --data_dir ./datasets/BasicMotions \
#    --data_class tsra \
#    --pattern TRAIN \
#    --val_pattern TEST \
#    --epochs 100 \
#    --lr 0.0005 \
#    --patch_size 8 \
#    --stride 4 \
#    --optimizer AdamW \
#    --d_model 768 \
#    --pos_encoding learnable \
#    --task classification \
#    --key_metric accuracy \
#    --gpu 5 \
#    --seed 42 \
#    --gpt_layers 6 \
#    --num_mamba_layers 2
#
#python src/main.py \
#    --output_dir experiments \
#    --comment "classification from Scratch" \
#    --name BasicMotions \
#    --records_file Classification_records.xls \
#    --data_dir ./datasets/BasicMotions \
#    --data_class tsra \
#    --pattern TRAIN \
#    --val_pattern TEST \
#    --epochs 100 \
#    --lr 0.0005 \
#    --patch_size 8 \
#    --stride 4 \
#    --optimizer AdamW \
#    --d_model 768 \
#    --pos_encoding learnable \
#    --task classification \
#    --key_metric accuracy \
#    --gpu 5 \
#    --seed 42 \
#    --gpt_layers 6 \
#    --num_mamba_layers 2
#
#python src/main.py \
#    --output_dir experiments \
#    --comment "classification from Scratch" \
#    --name CharacterTrajectories \
#    --records_file Classification_records.xls \
#    --data_dir ./datasets/CharacterTrajectories \
#    --data_class tsra \
#    --pattern TRAIN \
#    --val_pattern TEST \
#    --epochs 100 \
#    --lr 0.0005 \
#    --patch_size 8 \
#    --stride 4 \
#    --optimizer AdamW \
#    --d_model 768 \
#    --pos_encoding learnable \
#    --task classification \
#    --key_metric accuracy \
#    --gpu 5 \
#    --seed 42 \
#    --gpt_layers 6 \
#    --num_mamba_layers 2

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name Epilepsy \
    --records_file Classification_records.xls \
    --data_dir ./datasets/Epilepsy \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 100 \
    --lr 0.0005 \
    --patch_size 8 \
    --stride 4 \
    --optimizer AdamW \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --gpu 5 \
    --seed 42 \
    --gpt_layers 6 \
    --num_mamba_layers 2

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name ERing \
    --records_file Classification_records.xls \
    --data_dir ./datasets/ERing \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 100 \
    --lr 0.0005 \
    --patch_size 8 \
    --stride 4 \
    --optimizer AdamW \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --gpu 5 \
    --seed 42 \
    --gpt_layers 6 \
    --num_mamba_layers 2