# source /mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/bin/activate verl
python scripts/extract_eagle3_draft_from_fsdp.py \
    --actor-dir /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-eagle3-opd/fsdp/student-teacher-eagle3/loss-k3/train-mathcode16k_mbpp_user_prompt-update-accumulation-steps/global_step_2500/actor \
    --reference-draft-dir /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-eagle3-dflash_data-think/epoch_5_step_140000 \
    --target-dir /path/to/output_eagle3_draft \
    --overwrite