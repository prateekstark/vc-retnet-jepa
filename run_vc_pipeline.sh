#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"

# If Google Drive is mounted in Colab (/content/drive/MyDrive), save checkpoints & outputs directly to Drive
if [ -d "/content/drive/MyDrive" ]; then
    DRIVE_SAVE_DIR="${DRIVE_SAVE_DIR:-/content/drive/MyDrive/vc-retnet-jepa-runs}"
    mkdir -p "${DRIVE_SAVE_DIR}"
    CKPT_DIR="${CKPT_DIR:-${DRIVE_SAVE_DIR}/checkpoints_v2}"
    OUT_DIR="${OUT_DIR:-${DRIVE_SAVE_DIR}/outputs_v2}"
    RESULTS_FILE="${DRIVE_SAVE_DIR}/vc_results_v2.txt"
    echo "[Google Drive detected] Saving checkpoints, WAVs, and results directly to: ${DRIVE_SAVE_DIR}"
else
    CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/checkpoints_v2}"
    OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs_v2}"
    RESULTS_FILE="${REPO_DIR}/vc_results_v2.txt"
fi
PY="${PYTHON:-$(command -v python3 || command -v python)}"

WAV_A="${DATA_DIR}/cmu_us_bdl_arctic/wav"
WAV_B="${DATA_DIR}/cmu_us_clb_arctic/wav"
WAV_C="${DATA_DIR}/cmu_us_slt_arctic/wav"
HIFIGAN_CKPT="${DATA_DIR}/hifigan_16k_generator.ckpt"

EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-40}"
EPOCHS_CONVERT="${EPOCHS_CONVERT:-35}"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

echo "=== [0/5] Downloading Pretrained 16kHz HiFi-GAN & CMU Arctic Speakers (bdl, clb, slt) ==="
if [ ! -f "${HIFIGAN_CKPT}" ]; then
    echo "Downloading pretrained 16kHz HiFi-GAN vocoder..."
    curl -f -L -C - --retry 5 -s -S -o "${HIFIGAN_CKPT}" \
        "https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz/resolve/main/generator.ckpt"
fi
for spk in bdl clb slt; do
    if [ ! -d "${DATA_DIR}/cmu_us_${spk}_arctic/wav" ]; then
        echo "Downloading cmu_us_${spk}_arctic..."
        curl -f -L -C - --retry 5 -s -S -o "${DATA_DIR}/cmu_us_${spk}_arctic.tar.bz2" \
            "http://festvox.org/cmu_arctic/packed/cmu_us_${spk}_arctic.tar.bz2"
        tar -xjf "${DATA_DIR}/cmu_us_${spk}_arctic.tar.bz2" -C "${DATA_DIR}"
        rm -f "${DATA_DIR}/cmu_us_${spk}_arctic.tar.bz2"
    fi
done

echo "=== [1/5] Pretraining Scaled-Up Hybrid RetNet AE+JEPA (RMSNorm + SiLU + ConvPostNet) ==="
$PY train_hybrid.py --speaker a --wav-dir "${WAV_A}" --epochs "${EPOCHS_PRETRAIN}" --out "${CKPT_DIR}/hybrid_a_bdl.pt"
$PY train_hybrid.py --speaker b --wav-dir "${WAV_B}" --epochs "${EPOCHS_PRETRAIN}" --out "${CKPT_DIR}/hybrid_b_clb.pt"

echo "=== [2/5] Training B -> A Latent Converters (Unpaired, Paired, Hybrid AC) ==="
$PY train_converter.py --ckpt-a "${CKPT_DIR}/hybrid_a_bdl.pt" --ckpt-b "${CKPT_DIR}/hybrid_b_clb.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --mode unpaired  --epochs "${EPOCHS_CONVERT}" --out "${CKPT_DIR}/conv_unpaired.pt"
$PY train_converter.py --ckpt-a "${CKPT_DIR}/hybrid_a_bdl.pt" --ckpt-b "${CKPT_DIR}/hybrid_b_clb.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --mode paired    --epochs "${EPOCHS_CONVERT}" --out "${CKPT_DIR}/conv_paired.pt"
$PY train_converter.py --ckpt-a "${CKPT_DIR}/hybrid_a_bdl.pt" --ckpt-b "${CKPT_DIR}/hybrid_b_clb.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --mode hybrid_ac --epochs "${EPOCHS_CONVERT}" --out "${CKPT_DIR}/conv_hybrid_ac.pt"

echo "=== [3/5] Running Paper Ablations (1: No P_A Verification, 2: Pure AE without JEPA) ==="
$PY train_converter.py --ckpt-a "${CKPT_DIR}/hybrid_a_bdl.pt" --ckpt-b "${CKPT_DIR}/hybrid_b_clb.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --mode unpaired --lambda-jepa-verify 0.0 --epochs "${EPOCHS_CONVERT}" --out "${CKPT_DIR}/conv_unpaired_no_pa.pt"

$PY train_hybrid.py --speaker a --wav-dir "${WAV_A}" --lambda-jepa 0.0 --epochs "${EPOCHS_PRETRAIN}" --out "${CKPT_DIR}/pure_ae_a_bdl.pt"
$PY train_hybrid.py --speaker b --wav-dir "${WAV_B}" --lambda-jepa 0.0 --epochs "${EPOCHS_PRETRAIN}" --out "${CKPT_DIR}/pure_ae_b_clb.pt"
$PY train_converter.py --ckpt-a "${CKPT_DIR}/pure_ae_a_bdl.pt" --ckpt-b "${CKPT_DIR}/pure_ae_b_clb.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --mode hybrid_ac --lambda-jepa-verify 0.0 --epochs "${EPOCHS_CONVERT}" --out "${CKPT_DIR}/conv_pure_ae.pt"

echo "=== [4/5] Second Speaker Pair (slt -> bdl) ==="
$PY train_hybrid.py --speaker b --wav-dir "${WAV_C}" --epochs "${EPOCHS_PRETRAIN}" --out "${CKPT_DIR}/hybrid_c_slt.pt"
$PY train_converter.py --ckpt-a "${CKPT_DIR}/hybrid_a_bdl.pt" --ckpt-b "${CKPT_DIR}/hybrid_c_slt.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_C}" --mode hybrid_ac --epochs "${EPOCHS_CONVERT}" --out "${CKPT_DIR}/conv_slt2bdl_ac.pt"

echo "=== [5/5] Evaluating All Models & Synthesizing HiFi-GAN Audio (.wav) ==="
{
    echo "--- 1. Unpaired Cycle + P_A Verification (clb -> bdl) ---"
    $PY convert_voice.py --converter-ckpt "${CKPT_DIR}/conv_unpaired.pt"       --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --hifigan-ckpt "${HIFIGAN_CKPT}" --out-dir "${OUT_DIR}/samples_unpaired"
    echo "--- 2. Ablation: Unpaired Cycle WITHOUT P_A Verification (clb -> bdl) ---"
    $PY convert_voice.py --converter-ckpt "${CKPT_DIR}/conv_unpaired_no_pa.pt" --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --hifigan-ckpt "${HIFIGAN_CKPT}" --out-dir "${OUT_DIR}/samples_unpaired_no_pa"
    echo "--- 3. Paired Cross-Decode + P_A Verification (clb -> bdl) ---"
    $PY convert_voice.py --converter-ckpt "${CKPT_DIR}/conv_paired.pt"         --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --hifigan-ckpt "${HIFIGAN_CKPT}" --out-dir "${OUT_DIR}/samples_paired"
    echo "--- 4. Hybrid AC (Cycle + Paired + P_A Verification, clb -> bdl) ---"
    $PY convert_voice.py --converter-ckpt "${CKPT_DIR}/conv_hybrid_ac.pt"      --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --hifigan-ckpt "${HIFIGAN_CKPT}" --out-dir "${OUT_DIR}/samples_hybrid_ac"
    echo "--- 5. Ablation: Pure AE (lambda_jepa=0, clb -> bdl) ---"
    $PY convert_voice.py --converter-ckpt "${CKPT_DIR}/conv_pure_ae.pt"        --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_B}" --hifigan-ckpt "${HIFIGAN_CKPT}" --out-dir "${OUT_DIR}/samples_pure_ae"
    echo "--- 6. Second Speaker Pair: Hybrid AC (slt -> bdl) ---"
    $PY convert_voice.py --converter-ckpt "${CKPT_DIR}/conv_slt2bdl_ac.pt"     --wav-dir-a "${WAV_A}" --wav-dir-b "${WAV_C}" --hifigan-ckpt "${HIFIGAN_CKPT}" --out-dir "${OUT_DIR}/samples_slt2bdl_ac"
} | tee "${REPO_DIR}/vc_results_v2.txt"

if [ "${RESULTS_FILE}" != "${REPO_DIR}/vc_results_v2.txt" ]; then
    cp -f "${REPO_DIR}/vc_results_v2.txt" "${RESULTS_FILE}"
fi

echo "=== All Experiments Complete! ==="
echo "Checkpoints saved in: ${CKPT_DIR}"
echo "WAV samples saved in: ${OUT_DIR}"
echo "Metrics saved in:     ${RESULTS_FILE}"
