<div align="center">

# Awesome 2D Gaussian Splatting

---
**Survey Paper:** [Beyond Pixels and Neural Fields: A Survey of 2D Gaussian Splatting for Image and Video Representation and Compression](https://www.alphaxiv.org/pdf/2610.2d-gaussian-splatting-image-video-representation)

---

## Citation

If you find this list useful, please cite the survey it is based on:

```bibtex
@misc{wang20262dgssurvey,
	title={{Beyond Pixels and Neural Fields: A Survey of 2D Gaussian Splatting for Image and Video Representation and Compression}},
	author={Wang, Miaohui and Huang, Yanji and Wang, Dayong and Xiong, Jian and Chen, Mingkai and Li, Yanshan},
  howpublished={\url{https://www.alphaxiv.org/pdf/2610.2d-gaussian-splatting-image-video-representation}},
	year={2026}
}
```

---

---

**A curated list of papers on 2D Gaussian Splatting (2DGS) for image and video representation and compression**

![Awesome](https://img.shields.io/badge/Awesome-List-fc60a8?logo=awesomelists)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![Papers](https://img.shields.io/badge/Papers-44-blue)
![Updated](https://img.shields.io/badge/Updated-Oct%202026-orange)

*Beyond Pixels and Neural Fields: from transform codecs, to implicit neural representations, to explicit anisotropic 2D Gaussians.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Taxonomy](#taxonomy)
- [Background](#background)
- [Papers](#papers)
  - [1. Gaussian Allocation & Densification](#1-gaussian-allocation--densification)
  - [2. Quantization & Entropy Coding](#2-quantization--entropy-coding)
  - [3. Fast & Feed-Forward Encoding](#3-fast--feed-forward-encoding)
  - [4. Hybrid Implicit-Explicit Representations](#4-hybrid-implicit-explicit-representations)
  - [5. Video Representation & Compression](#5-video-representation--compression)
  - [6. Scaling & Downstream Applications](#6-scaling--downstream-applications)
- [Related Surveys](#related-surveys)
- [Datasets & Metrics](#datasets--metrics)
- [Comparison with Traditional and Neural Codecs](#comparison-with-traditional-and-neural-codecs)
- [Open Challenges](#open-challenges)
- [Citation](#citation)
- [Contributing](#contributing)

---

## Overview

Image and video compression has long been dominated by **pixel-based transform coding** (JPEG, JPEG2000, H.264/AVC, HEVC, VVC, AV1) and, more recently, by **implicit neural representations** (INRs) such as SIREN, COIN/COIN++ and NeRV. **2D Gaussian Splatting (2DGS)** is an emerging third paradigm: an image or video is represented as an **explicit set of anisotropic 2D Gaussians** that are rendered by differentiable rasterization.

| Paradigm | Representation | Strengths | Limitations |
|---|---|---|---|
| Pixel-based (JPEG, H.26x) | Regular pixel grid + transform coefficients | Mature, hardware-optimized, widely deployed | Tied to a discrete grid and predefined coding structures |
| Neural field (COIN, NeRV) | Coordinate-based MLP | Continuous, compact | Costly neural decoding and per-signal optimization |
| **2DGS** (GaussianImage, ...) | Explicit anisotropic Gaussians | Explicit, continuous, editable; 100-1000+ FPS rendering | Unordered, variable-cardinality set makes entropy modeling hard |

**Key observations (from the survey)**

- 2DGS gives compact representations and very fast decoding, but compression efficiency depends strongly on **Gaussian allocation**, **attribute coding**, and **learned components**.
- Encoding is the bottleneck: optimization-based fitting can take tens to hundreds of seconds per image. **Learned initialization and feed-forward methods** cut this substantially.
- Entropy modeling for unordered Gaussian sets, bitstream standardization, and evaluation beyond pixel fidelity remain open.

---

## Taxonomy

Methods are grouped by the **primary pipeline stage** they address. Categories are not mutually exclusive, so each method is given a primary category plus secondary tags.

| Tag | Category | Question it answers |
|---|---|---|
| **A** | Allocation & Densification | Where and how many Gaussians? |
| **Q** | Quantization & Entropy Coding | How many bits per Gaussian? |
| **F** | Fast & Feed-Forward Encoding | How fast can we encode? |
| **H** | Hybrid Implicit-Explicit | How to represent? |
| **V** | Video Extensions | How to exploit temporal redundancy? |
| **D** | Downstream Applications | What can we do beyond compression? |

---

## Background

### Gaussian primitive

Each 2D Gaussian is parameterized by position `mu` (2), covariance `Sigma` (3), color `c` (3) and opacity `o` (1), which gives 9 parameters per Gaussian (vs. 59 for a 3D Gaussian).

```
G_i(x) = exp( -1/2 (x - mu_i)^T  Sigma_i^{-1}  (x - mu_i) )
```

### Covariance parameterization

- **Rotation-scaling:** `Sigma = R(theta) S S^T R(theta)^T`. Intuitive, but ill-conditioned near isotropy and awkward to quantize.
- **Cholesky:** `Sigma = L L^T` with three scalars `(l11, l21, l22)`. No angular variable, cheap to evaluate, and quantization-friendly (used by GaussianImage).

### Accumulated-summation rasterization

2DGS needs no camera projection or depth sorting, so alpha blending is replaced by an order-independent weighted sum:

```
C(x) = sum_i  c_i * o_i * G_i(x)
```

Color and opacity merge into one weighted color `c~_i = c_i * o_i`, which gives **8 parameters per Gaussian**.

### Compression as rate-distortion optimization

```
min_X^  D(X, X^) + lambda * R(X^)
R_total = N_g * (b_pos + b_cov + b_col + b_op) + R_meta
```

Bitrate is controlled through the Gaussian count `N_g` and the per-attribute bit budgets.

---

## Papers

> **Legend:** `Primary` / secondary tags as in the [taxonomy](#taxonomy). arXiv links are given where available.

### 1. Gaussian Allocation & Densification

Initialization, density control, and structure-/entropy-guided allocation.

| Method | Venue | Tags | Core contribution |
|---|---|---|---|
| **GaussianImage** | ECCV 2024 | Q, A | Accumulated-summation 2DGS codec; residual vector quantization. *The foundational work.* |
| EigenGS | CVPR 2025 | A, F | Dataset-level PCA eigen-Gaussian basis for instant initialization |
| EllipssianNet | WACV 2026 | A, F | Feed-forward center/covariance maps; Voronoi-trained image-guided sampling |
| Image-GS | SIGGRAPH 2025 | A, F | Content-adaptive allocation; progressive error-guided growth |
| GaussianImage++ | AAAI 2026 | A, Q | Distortion-driven densification; learned step-size quantization |
| Spectrum-adaptive | ICME 2025 | A, Q | Spectral-complexity-guided Gaussian distribution |
| Structure-guided [[arXiv](https://arxiv.org/abs/2512.24018)] | arXiv 2025 | A, Q | Segmentation/SLIC-prior allocation with hybrid entropy signal |
| Contour-aware | ICCE 2026 | A | Contour-information-aware primitive placement |
| EA-GI | Signal Processing 2026 | A | Entropy-aware quadtree partitioning (allocation, not entropy coding) |
| LocoADC [[arXiv](https://arxiv.org/abs/2607.17896)] | arXiv 2026 | A | Locality-aware bidirectional density control; plug-and-play |

**Summary of reported gains** (each against its own baseline and protocol, so not directly comparable):

| Method | Allocation prior | Static / Dynamic | Reported gain |
|---|---|---|---|
| GaussianImage | None (random) | Static | Baseline |
| EigenGS | Eigen-Gaussians | Both | 41.8 dB @ 10K iters on FFHQ |
| EllipssianNet | Feed-forward parameter maps | Both | +2.1 dB @ 5K iters on Kodak |
| Image-GS | Saliency + error | Both | 10x faster optimization |
| GaussianImage++ | Reconstruction distortion | Dynamic | ~ +3 dB on Kodak |
| Spectrum-adaptive | Spectral complexity | Both | +1.20 dB on Kodak |
| Structure-guided | Gradient variance (SLIC) | Static | 43.44% BD-Rate saving on Kodak |
| EA-GI | Hybrid entropy | Static | +1.17 dB on Kodak |
| LocoADC | Error region + similarity | Dynamic | +3.02 dB on Kodak |
| Contour-aware | Segmentation prior | Static | Best at very low primitive counts |

---

### 2. Quantization & Entropy Coding

| Method | Venue | Tags | Core contribution | Reported gain |
|---|---|---|---|---|
| **GaussianImage** | ECCV 2024 | Q | FP16 position + b-bit asymmetric covariance + RVQ color | Baseline |
| GaussianImage++ | AAAI 2026 | Q | LSQ+ learned scalar quantizers per attribute | 25-147% BD-Rate (ablation) |
| Spectrum-adaptive | ICME 2025 | Q | DCT on position field + learnable integer quantization | Better RD vs. COIN/COIN++ |
| Structure-guided | arXiv 2025 | Q | Adaptive bitwidth quantization (ABQ) | 43.44% BD-Rate saving (Kodak) |
| CGVQ [[arXiv](https://arxiv.org/abs/2607.05667)] | SIGGRAPH 2026 | Q | K-means cluster-wise codebooks | 20% BPP saving |
| Dictionary-based | IEEE SPL 2025 | Q | Pre-trained dictionaries + ANS entropy coding | 17.96% (Kodak), 15.98% (DIV2K) BD-Rate |

**Take-aways:** attribute-specific quantization matters; cross-image redundancy (shared dictionaries, learned priors) is under-explored; and there is **no fully differentiable rate term** for Gaussian attributes yet.

---

### 3. Fast & Feed-Forward Encoding

| Method | Venue | Amortization | Per-image opt. | Encoding time (reported) |
|---|---|---|---|---|
| GaussianImage | ECCV 2024 | None | Full fitting is 50K iters; AIR's reproduction reduced it to 0.8K (Kodak) / 1K (DIV2K) | ~1463-2047 ms at the reduced iteration count (full fitting: 293-396 s @ A30, see the codec table below) |
| Instant-GI | ICCV 2025 | Learned/meta-learned init | Light refinement | ~191-503 ms (network init) |
| Fast-2DGS | WACV 2026 | Learned deep Gaussian prior | Light refinement | ~2 s to 40 dB on Kodak |
| AIR [[arXiv](https://arxiv.org/abs/2605.20820)] | arXiv 2026 | Full feed-forward (stage-wise residual, predict-optimize-distill) | None | ~157-300 ms |

- **Instant-GI:** position probability map, dithering + Delaunay, MLP attribute fields.
- **Fast-2DGS:** FiLM-conditioned Gaussian count, so one model covers budgets up to K_max = 100K.
- **AIR:** DINOv3 features, stage-wise residual prediction, image-adaptive quantizer. It has no differentiable rate estimate yet.

---

### 4. Hybrid Implicit-Explicit Representations

| Method | Venue | Core contribution | Key results (as reported) |
|---|---|---|---|
| PA-G2DS | IEEE TCSVT 2026 | Pixel-aligned grid (no position coding); learnable MLP renderer with IGDN; hyperprior / checkerboard / parallel / VQ variants | HP: 17.0% BD-Rate (Kodak), 28 ms enc / 7 ms dec @ GTX 1080Ti |
| SGI [[arXiv](https://arxiv.org/abs/2603.07789)] | arXiv 2026 | Seed-based structured Gaussians generated by shared MLPs; seed-level context entropy coding | Low-rate +3.94 dB vs. GaussianImage; -84.9% encoding time (FGF2) |

PA-G2DS variants:

| Variant | BD-Rate (Kodak) | Encode / Decode (GTX 1080Ti) |
|---|---|---|
| HP | 17.0% | 28 ms / 7 ms |
| CB | 5.1% | 39 ms / 10 ms |
| HP (Parallel) | 17.2% | 22 ms / 4 ms |
| VQ | 169.9% | 48 ms / 2 ms |

> BD-Rate figures are as printed in the survey's Table 5. The survey gives no anchor codec or sign convention, and the values sit awkwardly with its other text (e.g. VQ at 169.9% vs. CB described as "on par with BPG/HEVC/VVC" in Table 8). Check the original PA-G2DS paper before quoting them.
>
> Note on **Structure-guided**: the survey's Table 1 calls it "segmentation-prior allocation", while its Sec. 3.3 and Table 2 describe SLIC superpixels + Sobel gradients. Contour-aware is the one that uses object segmentation.

---

### 5. Video Representation & Compression

| Method | Venue | Temporal mechanism | Decoding speed | Reported position |
|---|---|---|---|---|
| HyperGS [[arXiv](https://arxiv.org/abs/2607.11500)] | arXiv 2026 | Feed-forward spatio-temporal Transformer (no per-video optimization) | 7.9 s per UVG video @ A100 | +3.34 dB over FastNeRV |
| GSVC | ACM NOSSDAV 2025 | I/P prediction; prune/add splats; dynamic key frames | 1583 FPS @ V100 | Competitive with VVC/AV1, DCVC-DC, HEM |
| NeuralVideo [[arXiv](https://arxiv.org/abs/2505.09324)] | arXiv 2025 | ROI-based selective P-frame coding for conferencing | Real-time @ RTX 3090 | 88% encoding speedup vs. per-frame fitting |
| DSGV | ACM MM Asia 2025 | Dynamic-static decomposition via optical flow | >900 FPS @ RTX 3090 | -78% storage vs. GSVC; +0.33 dB |
| GaussianVideo | CVPRW 2025 | Deformable canonical set; multi-plane encoder | 475-584 FPS @ A5000 | On par with NeRV family; 12.5x faster decoding |
| GSVR | CSCWD 2026 | Hybrid tri-plane/polynomial deformation field | 838.4 FPS @ A6000 | 30.22 dB UVG; +4.99 dB over NeRV |
| STGV [[arXiv](https://arxiv.org/abs/2604.10910)] | arXiv 2026 | 2D/3D hash-encoded deformation; key-frame init | 460 FPS @ RTX 3090 | +9.02 dB over NeRV |
| DeformVideo [[arXiv](https://arxiv.org/abs/2609.14129)] | arXiv 2026 | Multi-scale GOP; flow-guided deformation (4K) | 247.2 FPS @ RTX 5000 | Beats H.265/HEVC in LPIPS RD |
| P-GSVC | ACM MMSys 2026 | Layered base/enhancement splats; scalable coding | 1200 FPS @ H100 | +1.9 dB on UVG |
| D2GV-AR [[arXiv](https://arxiv.org/abs/2503.05600)] | arXiv 2026 | Neural-ODE deformation; Nyquist scale grouping; any-ratio pruning | 205 FPS @ RTX 3090 | 35.83 dB on UVG |

**Four strategies:** (1) temporal prediction and propagation, (2) dynamic-static decomposition, (3) deformable canonical sets, (4) scalable and feed-forward video coding.

---

### 6. Scaling & Downstream Applications

#### Large / ultra-high-resolution images

| Method | Venue | Core contribution | Main result (reported) |
|---|---|---|---|
| LIG | AAAI 2025 | Level-of-Gaussian hierarchical fitting; direct covariance optimization | Stable scaling to 3.5x10^7 Gaussians; 37.47 dB @ 9K, 51.81 dB @ 4K |
| SmartSplat | AAAI 2026 | Feature-aware sampling; scale-adaptive color init; tile-based training | Stable up to 3000x compression where baselines fail beyond 500x (parameter-based ratio, *not* an entropy-coded bitrate) |

#### Arbitrary-scale super-resolution

| Method | Venue | Core contribution |
|---|---|---|
| GaussianSR | AAAI 2025 | First 2DGS arbitrary-scale SR; Gaussians anchored to the LR pixel grid |
| GSASR | ICCV 2025 | Feed-forward image-conditioned Gaussians; scale-aware rasterization |
| ContinuousSR | ICLR 2026 | Single-pass continuous Gaussian field; deep Gaussian prior; 1 ms per scale |
| GSMSR | Pattern Recognition 2026 | Scale-aware cross-modal Transformer for multi-modal MRI |
| GS-STVSR [[arXiv](https://arxiv.org/abs/2604.18047)] | arXiv 2026 | Continuous spatio-temporal video SR via evolving Gaussians |

#### Editing and restoration

| Method | Venue | Core contribution |
|---|---|---|
| MiraGe [[arXiv](https://arxiv.org/abs/2410.01521)] | arXiv 2026 | Flat 3D Gaussians via mirror reflection; physics-based image editing |
| LL-GaussianImage [[arXiv](https://arxiv.org/abs/2601.15766)] | arXiv 2026 | Zero-shot low-light enhancement in the 2DGS domain |
| Fi-Gaussian [[arXiv](https://arxiv.org/abs/2606.16168)] | arXiv 2026 | Frequency-aware implicit Gaussian splatting for single-image dehazing |

#### Vision, vision-language and tokenization

| Method | Venue | Core contribution |
|---|---|---|
| GViT [[arXiv](https://arxiv.org/abs/2506.23532)] | arXiv 2025 | Learnable 2D Gaussians as compact, class-guided ViT inputs (76.9% top-1, matching the 86M-parameter backbone) |
| GaussianToken [[arXiv](https://arxiv.org/abs/2501.15619)] | arXiv 2025 | 2DGS image tokenizer with per-Gaussian vector quantization |
| GaussianVision | CVPR 2026 | Splat-aware CLIP stem; vision-language alignment from 2DGS (3-23.5x input compression) |
| GauSAM | NeurIPS 2025 | Contour-guided Gaussian probability fields with SAM for medical segmentation |

#### Domain-specific signals

| Method | Venue | Domain |
|---|---|---|
| GS-HSI | Chinese Optics Letters 2025 | Hyperspectral image compression (cross-band prior) |
| GSLR | CVPR 2026 | Multi-dimensional image recovery (2D+1D Gaussians in t-SVD) |
| CGH-GS | ECCV 2026 | Complex-valued primitives for computer-generated holography |

---

## Related Surveys

**General 3DGS**
- *A Survey on 3D Gaussian Splatting.* Chen and Wang. ACM Computing Surveys, 2026.
- *Recent Advances in 3D Gaussian Splatting.* Wu et al. Computational Visual Media, 2024.
- *3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities.* Bao et al. IEEE TCSVT, 2025.
- *A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation.* He et al. IEEE TPAMI, 2026.

**3DGS compression**
- *Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions.* Ali et al. IEEE TCSVT, 2026.
- *3DGS.zip: A Survey on 3D Gaussian Splatting Compression Methods.* Bagdasarian et al. Computer Graphics Forum, 2025.
- *SUCCESS-GS: Survey of Compactness and Compression for Efficient Static and Dynamic Gaussian Splatting.* Youn et al. [arXiv:2512.07197](https://arxiv.org/abs/2512.07197), 2025.

**Learned image and video compression**
- *Image and Video Compression With Neural Networks: A Review.* Ma et al. IEEE TCSVT, 2019.
- *Learning End-to-End Lossy Image Compression: A Benchmark.* Hu et al. IEEE TPAMI, 2021.
- *Advances in Video Compression System Using Deep Neural Network: A Review and Case Studies.* Ding et al. Proceedings of the IEEE, 2021.
- *End-to-End Neural Video Compression: A Review.* Gomes et al. IEEE OJCAS, 2025.

---

## Datasets & Metrics

### Datasets

| Dataset | Type | Used for |
|---|---|---|
| Kodak | Image (true-color, lossless) | Standard image compression benchmark |
| DIV2K | Image (high-res) | Diverse high-resolution evaluation |
| CLIC | Image (professional + mobile) | Learned compression evaluation |
| UVG | Video | Video compression (diverse motion) |
| DAVIS, Big Buck Bunny | Video | Video representation (NeRV-style comparisons) |
| STimage-1K4M | 9K histopathology | Scalability to ultra-high-resolution |
| FGF2 (Gaofen-2) | 4K satellite | Scalability to large images |
| DIV8K | 8K images | Ultra-high-resolution compression |

### Metrics

| Metric | Measures | Better |
|---|---|---|
| PSNR | Pixel fidelity | Higher |
| MS-SSIM | Structural similarity | Higher |
| LPIPS | Learned perceptual distance | Lower |
| VMAF | Learned video quality | Higher |
| BPP | Bits per pixel | Lower |
| BD-Rate / BD-PSNR | Average rate / quality difference between RD curves | Negative / positive |
| FPS | Decoding / rendering throughput | Higher |

---

## Comparison with Traditional and Neural Codecs

> Figures are **as reported by the cited papers** under heterogeneous protocols and hardware. This table summarizes trends and is **not a controlled benchmark**.

| Codec | Family | RD position (reported) | Encoding | Decoding |
|---|---|---|---|---|
| JPEG | Traditional | Baseline; weak MS-SSIM at low bpp | ~609 / 557 FPS (CPU) | ~615 / 515 FPS (CPU) |
| JPEG2000 | Traditional | Better PSNR than early 2DGS codecs | 3.4 FPS (CPU) | 4.3 / 3.9 FPS (CPU) |
| Balle2018 | Neural (learned) | Above GaussianImage / GI++ in PSNR | 21.2 / 16.5 / 13.5 FPS @ V100 | 18.8 / 15.2 FPS @ V100 |
| COIN | Neural (INR) | Below GaussianImage; very slow encoding | ~5x10^-4 FPS @ V100 | 166.3 / 93.7 FPS @ V100 |
| ELIC | Neural (learned) | Exceeds VVC by 7.8% BD-Rate | 42 ms @ TITAN Xp | 49 ms @ TITAN Xp |
| LALIC | Neural (learned) | Exceeds VVC by 14.8% BD-Rate | 274 ms @ RTX 4090 | 150 ms @ RTX 4090 |
| GaussianImage | 2DGS explicit | > COIN/COIN++; < JPEG2000, Balle2018 | 293-396 s @ A30 | ~1970 FPS |
| GaussianImage++ | 2DGS explicit | > GaussianImage; > JPEG at low bpp; gap to learned codecs | 338-576 s @ A30 | > GaussianImage |
| PA-G2DS-CB | 2DGS hybrid | On par with BPG/HEVC/VVC | 39 ms @ 1080Ti | 100 FPS @ 1080Ti |
| GSVC | 2DGS video | Comparable to VVC/AV1 on UVG | 197-222 s/frame @ V100 | 1583 FPS (1080p) @ V100 |
| DSGV | 2DGS video | +0.33 dB over GSVC; -78% storage | 154 s/frame @ RTX 3090 | >900 FPS @ RTX 3090 |
| NeuralVideo | 2DGS video (ROI) | Conferencing regime | 0.67 FPS @ RTX 3090 | Real-time @ RTX 3090 |

**Take-aways**

- Encoding, not decoding, is the main cost of optimization-based 2DGS.
- Explicit codecs are competitive with traditional codecs in some regimes but generally trail highly optimized learned codecs, especially in PSNR.
- Hybrid designs (learned decoders and entropy models) narrow the gap, at the price of decoder complexity.
- Hardware portability (CPU, mobile, FPGA/ASIC/NPU) is largely unstudied.

---

## Open Challenges

1. **Entropy modeling for unordered Gaussian sets.** Raster-order and neighborhood assumptions of hyperprior/context models do not transfer. Candidate directions: anchor-based contexts (ContextGS / HAC style), Morton/Hilbert/tile ordering, and set-to-sequence modeling.
2. **Bitstream standardization and rate control.** Today each method has its own bitstream. A reference codec with a standardized syntax, conformance streams, and a QP-like quality control is missing.
3. **Generative priors for extreme compression.** Diffusion-style priors have been explored for 3DGS (e.g. ExGS) but not for 2DGS. Hallucination may be unacceptable for medical or satellite imaging.
4. **Temporal coherence and streaming for video.** Long-sequence drift, random access, adaptive bitrate, error recovery, and encoding latency.
5. **Evaluation beyond pixel fidelity.** Perceptual (LPIPS, DISTS) and task-aware objectives, plus unified benchmarks (datasets, operating points, hardware, metrics).

---

## Citation

If you find this list useful, please cite the survey it is based on:

```bibtex
@misc{wang20262dgssurvey,
	title={{Beyond Pixels and Neural Fields: A Survey of 2D Gaussian Splatting for Image and Video Representation and Compression}},
	author={Wang, Miaohui and Huang, Yanji and Wang, Dayong and Xiong, Jian and Chen, Mingkai and Li, Yanshan},
  howpublished={\url{https://www.alphaxiv.org/pdf/2610.2d-gaussian-splatting-image-video-representation}},
	year={2026}
}
```

---

## Contributing

Contributions are welcome! To add a paper:

1. Fork this repository and add a row to the appropriate table, keeping the existing column format.
2. Include **method name, venue/year, tags, and a one-line core contribution**, plus a paper link if available.
3. Report numbers **as stated in the paper** and note the dataset/hardware, since protocols differ.
4. Open a pull request.

**Scope:** methods that use 2D Gaussian primitives as the *primary* representation of image or video signals, with emphasis on compression-related aspects. Papers that use 2D Gaussians only for novel view synthesis, 3D scene or surface reconstruction, SLAM, or similar 3D tasks, and INR-based compression without Gaussian primitives, are out of scope.

---

<div align="center">

*Last updated: October 2026*

</div>
