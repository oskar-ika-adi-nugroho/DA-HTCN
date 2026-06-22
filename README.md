# DHA-eGCN: Differential Hyperedge Attention-Enhanced Graph Convolution Network for Skeleton-Based Human Action Recognition

This repository provides the reference implementation of **DHA-eGCN**, a hybrid skeleton-based human action recognition framework published in **Sensors**.

DHA-eGCN is designed for recognizing human actions from 3D skeleton sequences. It combines **Differential Hyperedge Attention**, explicit **Graph Convolution Network** grounding, and **Multi-Scale Temporal Convolution** to preserve human body topology while capturing long-range spatiotemporal dependencies.

The implementation is built on top of the Hyperformer codebase and extends it with differential attention, masked and adaptive graph branches, full-depth graph modeling, and enriched multi-stream skeleton features.

## Paper

**DHA-eGCN: Differential Hyperedge Attention-Enhanced Graph Convolution Network for Skeleton-Based Human Action Recognition**  
Oskar Ika Adi Nugroho and Wen-Nung Lie  
*Sensors*, 2026, 26(12), 3932  
DOI: [10.3390/s26123932](https://doi.org/10.3390/s26123932)  
Paper: [https://www.mdpi.com/1424-8220/26/12/3932](https://www.mdpi.com/1424-8220/26/12/3932)

Please cite the paper if you use this repository, the model design, or the reported experimental results.

## Abstract-Level Summary

Skeleton-based human action recognition requires a model that can preserve the local kinematic structure of the human body while also capturing long-range dependencies across joints and time. Conventional GCN-based methods provide strong skeletal topology priors, but they can be limited by local aggregation. Attention-based methods capture global interactions, but they may also attend to noisy or weakly constrained correlations.

DHA-eGCN addresses this problem by combining topology-aware Differential Hyperedge Attention with graph convolution and multi-scale temporal convolution. The attention module uses hop-distance relative positional encoding, hyperedge context tokens from joint-to-part pooling, and a two-branch differential attention mechanism to suppress shared noisy correlations. The graph branch provides additional spatial grounding through masked or masked-adaptive GCN variants.

The paper evaluates DHA-eGCN on **NTU RGB+D 60**, **NTU RGB+D 120**, and **Northwestern-UCLA**. The best reported configuration achieves **93.7% / 97.0%** on NTU RGB+D 60 X-Sub / X-View, **90.9% / 91.9%** on NTU RGB+D 120 X-Sub / X-Set, and **97.6%** on Northwestern-UCLA.

## Main Contributions

- **Differential Hyperedge Attention (DHA)** combines topology-aware attention, hyperedge context, and differential attention for selective spatial modeling.
- **MGCN and MAGCN graph branches** strengthen local skeleton topology modeling with masked graph convolution and optional sample-adaptive adjacency.
- **Partial-GCN and full-GCN configurations** allow the graph branch to be applied only in early layers or across all ten backbone layers.
- **RICH4 and RICH5 multi-stream representations** provide complementary spatial and temporal skeleton features.
- **Multi-model ensemble selection** allows different streams to use different graph variants before late fusion.
- **Missing-joint robustness analysis** shows that DHA-eMAGCN full-GCN is more stable than Hyperformer under random testing-time joint masking.

## Method Overview

Given an input skeleton sequence:

$$
X \in \mathbb{R}^{N \times C \times T \times V \times M}
$$

where:

- $N$ is the batch size,
- $C$ is the coordinate channel,
- $T$ is the number of frames,
- $V$ is the number of joints,
- $M$ is the number of persons.

DHA-eGCN follows a hierarchical spatiotemporal pipeline:

1. **Input normalization and reshaping**  
   The skeleton tensor is normalized over person-joint-channel dimensions. The person dimension is merged into the batch dimension during backbone processing.

2. **Spatial modeling with DHA**  
   DHA computes structure-aware attention using hop-based relative positional encoding, hyperedge context tokens, and two-branch differential attention.

3. **Graph-enhanced spatial modeling**  
   A GCN branch is added to reinforce local topology-aware aggregation. The branch can be implemented as MGCN or MAGCN.

4. **Temporal modeling with MSTCN**  
   Multi-scale temporal convolution captures short-term and long-range motion patterns through parallel temporal branches.

5. **Global pooling and classification**  
   Features are globally pooled over time and joints. For multi-person inputs, person-level features are averaged before the final classifier.

## Architecture

### DHA-eGCN Block

Each DHA-eGCN block contains:

- a spatial module based on DHA,
- an optional MGCN or MAGCN branch,
- a multi-scale temporal convolution module,
- residual connections,
- stochastic depth through DropPath.

The paper uses a 10-layer backbone. Temporal downsampling is applied at layers 5 and 8.

### Differential Hyperedge Attention

DHA contains three key components.

#### 1. Hop-Based Relative Positional Encoding

Shortest-path hop distances on the physical skeleton graph are used to inject topology-aware relative positional encoding into the attention logits. This keeps attention aligned with the human body structure.

#### 2. Hyperedge Tokens from Joint-to-Part Pooling

Joint features are pooled into body-part-level tokens, such as torso, left arm, right arm, left leg, and right leg. These tokens are broadcast back to their corresponding joints and used as hyperedge context in attention computation.

#### 3. Differential Attention

DHA computes two structure-aware attention maps. The final attention map is produced by subtracting the second map from the first using a learnable coefficient $\lambda$:

$$
S = S^{(1)} - \lambda S^{(2)}
$$

This design is intended to suppress shared noisy correlations and preserve more discriminative interactions.

## Graph Branch Variants

### DHA-eMGCN

DHA-eMGCN uses **Masked GCN**. It applies learnable edge-importance masks on top of fixed skeleton adjacency partitions. This preserves the physical skeleton prior while allowing the model to emphasize task-relevant edges.

### DHA-eMAGCN

DHA-eMAGCN uses **Masked and Adaptive GCN**. It extends MGCN with a sample-dependent adaptive adjacency term. This allows the graph branch to capture action-conditioned relations beyond fixed physical bones.

## Graph Branch Placement

### Partial-GCN

The graph branch is applied only in layers 1 to 4. This setting provides early spatial grounding while keeping deeper layers mainly attention-based.

### Full-GCN

The graph branch is applied in all 10 layers. This keeps graph-based spatial reasoning active throughout the backbone. In the paper, the full-GCN setting gives the best overall performance.

## Multi-Stream Skeleton Features

DHA-eGCN supports both standard and enriched input streams.

### Standard 4-Stream Setting

- Joint
- Bone
- Joint Motion
- Bone Motion

### RICH4 Setting

- **J**: Joint feature
- **E**: Edge feature
- **S**: Surface feature
- **M**: Motion feature

### RICH5 Setting

- **J**: Joint feature
- **E**: Edge feature
- **S**: Surface feature
- **M**: Motion feature
- **A**: Acceleration-like feature

Each stream is processed by a dedicated DHA-eGCN model. The final prediction is obtained through weighted score-level late fusion.

## Figures

### Figure 1. Overall DHA-eGCN Architecture

![Figure 1](README_assets/Figure1.png)

**Figure 1.** Overall architecture of DHA-eGCN. The model processes skeleton sequences through normalization, stacked DHA-eGCN blocks, global average pooling, and a final classifier. The partial-GCN setting applies the GCN branch only in early layers, while the full-GCN setting applies it in all layers.

### Figure 2. DHA-eGCN Block

![Figure 2](README_assets/Figure2.png)

**Figure 2.** DHA-eGCN block. The spatial module combines DHA with an optional MGCN or MAGCN branch. The temporal module uses multi-scale temporal convolution.

### Figure 3. Projection and Splitting for Differential Attention

![Figure 3](README_assets/Figure3.png)

**Figure 3.** Input features are projected into query, key, and value tensors. Query and key tensors are split into two branches, producing $Q_1$, $Q_2$, $K_1$, and $K_2$, while $V$ is shared.

### Figure 4. Differential Hyperedge Attention

![Figure 4](README_assets/Figure4.png)

**Figure 4.** DHA computes two structure-aware attention maps using joint-to-joint relations, hop-based RPE, joint-to-hyperedge interactions, and hyperedge-derived bias. The second map is subtracted from the first using the learnable coefficient $\lambda$.

### Figure 5. Multi-Stream RICH5 Framework

![Figure 5](README_assets/Figure5.png)

**Figure 5.** RICH5 decomposes the skeleton sequence into five complementary streams: Joint, Edge, Surface, Motion, and Acceleration. Each stream is processed independently before late fusion.

### Figure 6. RICH5 Feature Decomposition

![Figure 6](README_assets/Figure6.png)

**Figure 6.** The five RICH5 streams encode first-order spatial features, second-order edge features, third-order surface features, motion features, and acceleration-like temporal features.

## Experimental Settings Reported in the Paper

The paper reports the following main training settings:

- Framework: PyTorch
- Epochs: 140
- Loss: cross-entropy
- Initial learning rate: 0.025
- Learning rate decay: factor of 0.1 at epochs 110 and 120
- Batch size: 32 for NTU RGB+D 60 and NTU RGB+D 120
- Input sequence length: 64 frames
- Backbone depth: 10 layers
- Temporal downsampling: layers 5 and 8
- GPU used in the reported experiments: NVIDIA GeForce RTX 4090

## Datasets

The paper evaluates DHA-eGCN on three public skeleton-based action recognition benchmarks.

| Dataset | Protocols | Notes |
|---|---|---|
| NTU RGB+D 60 | X-Sub, X-View | 60 action classes, 56,880 skeleton sequences |
| NTU RGB+D 120 | X-Sub, X-Set | 120 action classes, 113,945 skeleton sequences |
| Northwestern-UCLA | Cross-View | 10 action classes, cross-view evaluation |

Users should obtain each dataset from its official provider and follow the corresponding license and access terms.

## Main Results

### Comparison with State-of-the-Art Methods

| Method | Streams | NTU60 X-Sub | NTU60 X-View | NTU120 X-Sub | NTU120 X-Set | NW-UCLA |
|---|---:|---:|---:|---:|---:|---:|
| Hyperformer | 4 | 92.9 | 96.5 | 89.9 | 91.3 | 96.7 |
| SelfGCN | 4 | 93.1 | 96.6 | 89.4 | 91.0 | 96.8 |
| SkateFormer | 4 | 93.5 | 97.8 | 89.8 | 91.4 | 98.3 |
| DHA-eGCN, RICH4 full-GCN | 4 | **93.7** | 97.0 | **90.9** | **91.9** | 97.6 |

### Progressive Ablation on NTU RGB+D 60

| Backbone | Graph Placement | Streams | GCN Model | X-Sub | X-View |
|---|---|---|---|---:|---:|
| Hyperformer | N/A | STD4 | N/A | 92.8 | 96.5 |
| DHA-eMGCN | Partial | STD4 | MGCN | 93.1 | 96.8 |
| DHA-eMAGCN | Partial | STD4 | MAGCN | 93.2 | 96.7 |
| DHA-eMGCN | Full | STD4 | MGCN | 93.4 | 96.8 |
| DHA-eMAGCN | Full | STD4 | MAGCN | 93.4 | 96.8 |
| DHA-eMAGCN | Full | RICH5 | MAGCN | 93.6 | 96.9 |
| DHA-eGCN | Full | RICH4 | Multi-model selection | **93.7** | **97.0** |
| DHA-eGCN | Full | RICH5 | Multi-model selection | **93.7** | **97.0** |

## Computational Complexity

| Method | Stream | Params | GFLOPs | Inference Time | X-Sub | X-View |
|---|---:|---:|---:|---:|---:|---:|
| DHA-eMGCN partial | Joint | 9.58M | 17.81 | 9.06 ms/sample | 91.5 | 95.4 |
| DHA-eMAGCN partial | Joint | 9.73M | 17.82 | 9.34 ms/sample | 91.6 | 95.5 |
| DHA-eMGCN full-GCN | Joint | 11.48M | 20.84 | 10.17 ms/sample | 91.9 | 95.6 |
| DHA-eMAGCN full-GCN | Joint | 11.96M | 20.86 | 10.94 ms/sample | 92.0 | 95.6 |
| DHA-eGCN, RICH4 full-GCN | 4 | 46.88M | 83.40 | 42.22 ms/sample | 93.7 | 97.0 |

The single-stream setting is more suitable for lower-cost deployment. The RICH4 full-GCN ensemble is preferable when recognition accuracy is prioritized.

## Missing-Joint Robustness

The paper also evaluates robustness under testing-time random joint masking on NTU RGB+D 60 X-Sub using the joint stream. The models are trained with clean skeleton sequences and evaluated after randomly masking 10%, 20%, or 30% of joints by setting their 3D coordinates to zero.

| Model | Clean | Drop 10% | Drop 20% | Drop 30% |
|---|---:|---:|---:|---:|
| Hyperformer | 90.7 | 85.7 | 81.5 | 72.6 |
| DHA-eMAGCN full-GCN | **92.0** | **88.0** | **84.6** | **77.4** |

These results show that DHA-eMAGCN full-GCN is more robust to incomplete skeleton inputs, and the performance gap increases as the missing-joint ratio becomes larger.

## Notes on Ensemble Selection

The paper reports a post-training multi-model ensemble analysis in which each stream can select either MGCN or MAGCN before late fusion. For RICH4, there are $2^4 = 16$ possible combinations. For RICH5, there are $2^5 = 32$ possible combinations.

The NTU RGB+D 60 benchmark does not provide an official validation split. Therefore, the paper interprets the 93.7% NTU RGB+D 60 result as an analysis of stream and model complementarity. For stricter deployment, the ensemble configuration should be selected using a validation subset split from the training set and then fixed before final testing.

## Difference from Hyperformer

This repository is derived from Hyperformer, but introduces several key extensions:

1. **Differential attention**  
   Hyperformer-style attention is extended into a two-branch differential mechanism for attention noise suppression.

2. **Explicit GCN branch**  
   DHA-eGCN adds graph convolution to complement attention-based global modeling with local topology grounding.

3. **MGCN and MAGCN variants**  
   The graph branch supports masked fixed-topology GCN and masked-adaptive GCN.

4. **Full-depth graph modeling**  
   The graph branch can be applied across all 10 layers, not only in early layers.

5. **RICH4 and RICH5 feature streams**  
   Enriched spatial and temporal skeleton features are supported for stronger stream complementarity.

6. **Stream-wise model selection**  
   Different streams may use different graph variants before score-level late fusion.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{nugroho2026dhaegcn,
  title   = {DHA-eGCN: Differential Hyperedge Attention-Enhanced Graph Convolution Network for Skeleton-Based Human Action Recognition},
  author  = {Nugroho, Oskar Ika Adi and Lie, Wen-Nung},
  journal = {Sensors},
  volume  = {26},
  number  = {12},
  pages   = {3932},
  year    = {2026},
  doi     = {10.3390/s26123932},
  url     = {https://www.mdpi.com/1424-8220/26/12/3932}
}
```

## Acknowledgement

This implementation is built upon the Hyperformer codebase. We thank the authors of Hyperformer and related skeleton-based action recognition methods for making their work publicly available.
