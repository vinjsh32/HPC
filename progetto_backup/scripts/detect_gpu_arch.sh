#!/usr/bin/env bash
# Rileva la compute capability della prima GPU tramite nvidia-smi
# e stampa la corrispondente opzione -gencode per NVCC.
set -e
if command -v nvidia-smi >/dev/null 2>&1; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    cap=${cap//./}
    echo "-gencode arch=compute_${cap},code=sm_${cap}"
else
    # fallback di default se nvidia-smi non è disponibile
    echo "-gencode arch=compute_52,code=sm_52"
fi
