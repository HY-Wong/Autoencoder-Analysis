# Autoencoder-Analysis

This repository contains 8 analysis scripts (analysis-{model}.py) to evaluate the reconstruction quality of various autoencoders with different latent shapes on the ImageNet-1K validation set. The evaluations measure pixel- and frequency-space reconstruction loss, perceptual similarity (LPIPS), and feature similarity using pretrained [CLIP](https://github.com/openai/CLIP) and [SAM](https://github.com/facebookresearch/segment-anything). The scripts are designed to run in distributed settings and generate reconstruction visualizations.

## Sample Visualization

Below are sample batches showing ground-truth and reconstructed images:

| Ground Truth | Reconstruction (VA-VAE, f16c32) |
|:-------------:|:------------------------------:|
| <img src="images/ground-truth.png" width="400"> | <img src="images/recon_va-vae-f16c32.png" width="400"> |

> **Note:** The reconstructed outputs tend to lack sharp textures and fine details, particularly in regions dominated by high-frequency information such as text or small faces.

## Model Overview

Each analysis script (`analysis-{model}.py`) evaluates reconstruction quality for a specific autoencoder model.  
To run successfully, each script must be placed inside the **corresponding original model repository** (e.g., `analysis-ldm.py` inside the [latent-diffusion](https://github.com/CompVis/latent-diffusion) repo).

---

### 1D-Tokenizer
> 📄 *NeurIPS 2024*  
> **Image Tokenization with Only 32 Tokens for Both Reconstruction and Generation**  
> [GitHub Repository](https://github.com/bytedance/1d-tokenizer)

| Environment | Script |
|--------------|---------|
| `conda activate var` | `analysis-1d-tokenizer.py` |

---

### DiT
> 📄 *ICCV 2023*  
> **Scalable Diffusion Models with Transformers**  
> [GitHub Repository](https://github.com/facebookresearch/DiT)

| Environment | Script |
|--------------|---------|
| `conda activate var` | `analysis-DiT.py` |

---

### EfficientViT
> 📄 *ICCV 2023*  
> **Efficient Vision Foundation Models for High-Resolution Generation and Perception**  
> [GitHub Repository](https://github.com/mit-han-lab/efficientvit)

| Environment | Script |
|--------------|---------|
| `conda activate var` | `analysis-efficientvit.py` |

---

### Latent Diffusion (LDM)
> 📄 *CVPR 2022*  
> **High-Resolution Image Synthesis with Latent Diffusion Models**  
> [GitHub Repository](https://github.com/CompVis/latent-diffusion)

| Environment | Script |
|--------------|---------|
| `conda activate ldm` | `analysis-ldm.py` |

#### Checkpoints
Pretrained weights are available from the official repositories:  
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)  
- [Taming Transformers](https://github.com/CompVis/taming-transformers)

**VQ-VAEs**
- [`vq-f16c8v16384`](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/)
- [`vq-f16c256v16384`](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/)
- [`vq-f4c3v8192`](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip)
- [`vq-f8c4v16384`](https://ommer-lab.com/files/latent-diffusion/vq-f8.zip)
- [`vq-f8c4v256`](https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip)
- [`vq-f16c8v16384`](https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1)

**KL-VAEs**
- [`kl-f4c3`](https://ommer-lab.com/files/latent-diffusion/kl-f4.zip)
- [`kl-f8c4`](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)
- [`kl-f16c16`](https://ommer-lab.com/files/latent-diffusion/kl-f16.zip)
- [`kl-f32c64`](https://ommer-lab.com/files/latent-diffusion/kl-f32.zip)

---

### MAR
> 📄 *NeurIPS 2024*  
> **Autoregressive Image Generation without Vector Quantization**  
> [GitHub Repository](https://github.com/LTH14/mar)

| Environment | Script |
|--------------|---------|
| `conda activate ldm` | `analysis-mar.py` |

---

### RQ-VAE Transformer
> 📄 *CVPR 2022*  
> **Autoregressive Image Generation using Residual Quantization**  
> [GitHub Repository](https://github.com/kakaobrain/rq-vae-transformer)

| Environment | Script |
|--------------|---------|
| `conda activate var` | `analysis-rq-vae.py` |

---

### VA-VAE
> 📄 *CVPR 2025*  
> **Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models**  
> [GitHub Repository](https://github.com/hustvl/LightningDiT)

| Environment | Script |
|--------------|---------|
| `conda activate ldm` | `analysis-va-vae.py` |

---

### VAR
> 📄 *NeurIPS 2024*  
> **Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction**  
> [GitHub Repository](https://github.com/FoundationVision/VAR)

| Environment | Script |
|--------------|---------|
| `conda activate var` | `analysis-var.py` |

## Distributed Settings

Run the evaluation in a distributed setting using `torchrun`.

```python
torchrun \
  --nnodes=2 \                # number of nodes
  --nproc_per_node=2 \        # number of GPUs per node
  --node_rank=0 \             # rank of the current node (0 for master)
  --master_addr=10.0.0.1 \    
  --master_port=29500 \    
  analysis-{model}.py \       # replace with the specific model script
  --data_path </path/to/imagenet-1k> \
  --batch_size 64 \
  --resos 256
```

## Dataset

<details>
<summary>ImageNet Structure</summary>

```
/path/to/imagenet-1k/
    train/
        n01440764/
            *.JPEG
        n01443537/
            *.JPEG
    val/
        n01440764/
            ILSVRC2012_val_00000293.JPEG ...
        n01443537/
            ILSVRC2012_val_00000236.JPEG ...
```

</details>
