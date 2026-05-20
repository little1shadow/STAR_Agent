# STAR-Agent 外部下游工具接入说明

本文档说明如何接入 `tetra3rs` 和 `ASTRiDE`，以及如何用 `tetra3rs` 为真实 clean-data 生成 star pseudo mask。

## 1. 工具分工

```text
tetra3rs:
  用于星点质心提取、star pseudo mask 生成、后续 plate solving / star matching。

ASTRiDE:
  用于 short-streak target / 卫星条纹目标检测，不用于 star mask。

LOST:
  后续作为 star tracker / plate solving 对照工具接入，本轮暂不作为真实 clean star mask 主工具。
```

## 2. 为什么 tetra3rs 可以生成 star mask

`tetra3rs` 原生输出的是星点 centroid，不是像素级 mask。

STAR-Agent 当前做法是：

```text
real clean image
  -> tetra3rs.extract_centroids
  -> centroids.json
  -> centroid 周围局部阈值 + PSF 半径
  -> star_mask.png
  -> background_mask.png
  -> manifest_tetra3rs_pseudo.jsonl
```

因此生成的 mask 是 pseudo label，不是人工 GT。

## 3. 单张图测试 tetra3rs

如果当前环境已经安装 `tetra3rs`：

```bash
cd STAR_Agent

python scripts/downstream/run_tetra3rs.py \
  --image data/clean/real_selected_v001/images/frame_0120_2018206192942.png \
  --output_dir runs/downstream/tetra3rs/frame_0120
```

如果从 `agenticir` 主环境调用外部环境：

```bash
cd STAR_Agent

conda run -n star_downstream python scripts/downstream/run_tetra3rs.py \
  --image data/clean/real_selected_v001/images/frame_0120_2018206192942.png \
  --output_dir runs/downstream/tetra3rs/frame_0120
```

输出：

```text
runs/downstream/tetra3rs/frame_0120/star_mask.png
runs/downstream/tetra3rs/frame_0120/background_mask.png
runs/downstream/tetra3rs/frame_0120/valid_mask.png
runs/downstream/tetra3rs/frame_0120/centroids.json
runs/downstream/tetra3rs/frame_0120/metrics.json
```

## 4. 批量生成真实 clean-data star mask

如果真实数据结构是：

```text
data/clean/real_selected_v001/images/*.png
```

运行：

```bash
cd STAR_Agent

conda run -n star_downstream python scripts/downstream/build_real_clean_tetra3rs_masks.py \
  --clean_root data/clean/real_selected_v001 \
  --overwrite
```

如果现在多了一层目录，例如：

```text
data/clean/real_selected_v001/images/clean_data/*.png
```

运行：

```bash
cd STAR_Agent

conda run -n star_downstream python scripts/downstream/build_real_clean_tetra3rs_masks.py \
  --clean_root data/clean/real_selected_v001 \
  --recursive \
  --overwrite
```

批量输出：

```text
data/clean/real_selected_v001/masks/star_pseudo_tetra3rs/
data/clean/real_selected_v001/masks/background_pseudo_tetra3rs/
data/clean/real_selected_v001/masks/valid_pseudo_tetra3rs/
data/clean/real_selected_v001/labels/stars_tetra3rs/
data/clean/real_selected_v001/labels/downstream_proxy_tetra3rs/
data/clean/real_selected_v001/labels/tetra3rs_metrics/
data/clean/real_selected_v001/manifest_tetra3rs_pseudo.jsonl
```

## 5. 调阈值

配置文件：

```text
configs/downstream/star_matching/tetra3rs.yaml
```

如果星点漏检：

```text
降低 centroid_extraction.sigma_threshold
增大 mask.radius_px
降低 mask.local_threshold_sigma
```

如果噪声/坏点误检成星点：

```text
增大 centroid_extraction.sigma_threshold
增大 centroid_extraction.min_pixels
减小 centroid_extraction.max_elongation
```

## 6. ASTRiDE 单张图测试

ASTRiDE 用于条纹目标检测：

```bash
cd STAR_Agent

conda run -n star_astride python scripts/downstream/run_astride.py \
  --image data/clean/real_selected_v001/images/frame_0120_2018206192942.png \
  --output_dir runs/downstream/astride/frame_0120
```

输出：

```text
runs/downstream/astride/frame_0120/streak_mask.png
runs/downstream/astride/frame_0120/streaks.json
runs/downstream/astride/frame_0120/astride_summary.json
runs/downstream/astride/frame_0120/input_for_astride.fits
```

## 7. agenticir 主环境调用方式

Python adapter：

```python
from star_agent.downstream.star_matching.tetra3rs_adapter import run_tetra3rs_star_mask

metrics = run_tetra3rs_star_mask(
    image_path="data/clean/real_selected_v001/images/example.png",
    output_dir="runs/downstream/tetra3rs/example",
    conda_env="star_downstream",
)
```

ASTRiDE：

```python
from star_agent.downstream.target_detection.astride_adapter import run_astride_streak_detection

summary = run_astride_streak_detection(
    image_path="data/clean/real_selected_v001/images/example.png",
    output_dir="runs/downstream/astride/example",
    conda_env="star_astride",
)
```
