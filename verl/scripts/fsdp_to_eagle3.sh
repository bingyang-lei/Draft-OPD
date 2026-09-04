
step_list=(4000 5000 5304)

BASE_DIR=""
REF_DRAFT_DIR=""

for step in "${step_list[@]}"; do
    echo "Processing global_step_${step}..."

    python scripts/extract_eagle3_draft_from_fsdp.py \
        --actor-dir "${BASE_DIR}/global_step_${step}/actor" \
        --reference-draft-dir "${REF_DRAFT_DIR}" \
        --target-dir "${BASE_DIR}/global_step_${step}/draft" \
        --overwrite
    sleep 1
done