# Hybrid Split-Latent RetNet AE+JEPA for Voice Conversion ($B \to A$)

Implements the **Encoder + Decoder ($\mathcal{L}_{\text{recon}} + \lambda \, \mathcal{L}_{\text{JEPA}}$)** architecture with a **Split-Latent RetNet** backbone and a **$B \to A$ Latent Space Converter** verified by Target Speaker $A$'s frozen Decoder $D_A$ and JEPA Predictor $P_A$.

## Architecture Overview

1. **Stage 1 — Hybrid Split-Latent Pretraining (`train_hybrid.py`):**
   - **Split-Latent Encoder $E$**: Extracts an intermediate speaker/timbre style vector $z_{\text{style}}$ at mid-depth and a high-level semantic token sequence $z_{\text{sem}}$ at the top layer.
   - **JEPA Predictor $P$ + EMA Target Encoder $\bar{E}$**: Applies $\mathcal{L}_{\text{JEPA}}$ and variance anti-collapse purely on $z_{\text{sem}}$.
   - **FiLM RetNet Decoder $D$**: Reconstructs the 80-bin log-mel spectrogram $\hat{y} = D(z_{\text{sem}}, z_{\text{style}})$.
   - **Joint Objective**:
     $$\mathcal{L} = \lambda_{\text{recon}} \mathcal{L}_{\text{recon}}(x, \hat{y}) + \lambda_{\text{jepa}} \mathcal{L}_{\text{JEPA}}(z_{\text{sem}}) + \lambda_{\text{var}} \mathcal{L}_{\text{var}}(z_{\text{sem}})$$

2. **Stage 2 — $B \to A$ Latent Space Converter (`train_converter.py`):**
   - Freezes $(E_A, D_A, P_A)$ and $(E_B, D_B, P_B)$, and trains $T_{B \to A}: \mathcal{Z}_B \to \mathcal{Z}_A$ (plus $T_{A \to B}$ for cycle mode):
     - **`--mode unpaired`**: Non-parallel cycle consistency ($B \to A \to B$ and $A \to B \to A$) + **Frozen $P_A$ Manifold Verification** + **Content Self-Similarity** preservation.
     - **`--mode paired`**: Paired latent alignment $T_{B \to A}(E_B(x_B)) \to E_A(x_A)$ + **Cross-Decode Loss** $\mathcal{L}_{\text{recon}}(D_A(T_{B \to A}(E_B(x_B))), x_A)$ + $P_A$ verification.
     - **`--mode hybrid_ac`**: Combines unpaired cycle + paired cross-decode + $P_A$ manifold verification.

3. **Stage 3 — Evaluation & Audio Synthesis (`convert_voice.py`):**
   - Evaluates Mel L1 Distance, Latent Cosine Similarity to Target Speaker $A$, and $P_A$ Manifold Error.
   - Synthesizes playable `.wav` triplets (`source_b.wav`, `target_a_gt.wav`, `converted_b_to_a.wav`).

## Running the Full Experiment

Once the DCASE GPU run finishes, launch the complete CMU Arctic Voice Conversion experiment:

```bash
bash run_vc_pipeline.sh
```
