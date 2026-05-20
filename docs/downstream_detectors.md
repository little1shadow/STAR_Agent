# STAR-Agent 下游检测器补充方案

本文档对应 `docs/star_agent_construct.md` 第 8 节“下游任务反馈”。

## 1. 为什么现在要先补下游检测器

如果只生成 synthetic clean 和 degradation，还只能训练 restoration 本身。STAR-Agent 的论文核心不是单纯追求 PSNR，而是强调：

```text
restoration 是否保护了星点、目标和下游任务能力。
```

因此需要先补一套轻量下游检测器，用来产生：

```text
star_mask / target_mask / background_mask
star_count / target_candidate_count
star_snr / target_snr
target_confidence
online downstream proxy features
real clean pseudo labels
```

这些信息后续会进入：

```text
1. 真实 clean 的伪标签生成
2. executor 的区域加权 loss
3. Star-DepictQA-Lite 的 risk label / risk head
4. Policy Net 的 state feature
5. offline reward 和 downstream metric
```

## 2. 当前第一阶段补哪些 detector

根据 construct 文档，第一阶段先补两个下游任务：

```text
1. 星点检测 / 质心定位
2. 小目标检测
```

当前已经补入三个轻量 detector：

```text
star_detection/blob/detector.py
  星点 blob detector，用于 star pseudo mask、星点数量、SNR、FWHM proxy。

target_detection/log_blob_detector/detector.py
  点状弱目标 detector，对应 point_blob target。

target_detection/streak_detector/detector.py
  短条纹目标 detector，对应 short_streak target。
```

暂时不先接 LOST / tetra3rs / astrometry.net 的原因：

```text
1. 它们依赖星表、相机参数、plate solving 配置，工程链更长。
2. 当前最急的是先得到真实 clean 的 pseudo star/target/background mask。
3. Policy Net 第一阶段需要的是在线 proxy，不一定一开始就要完整 plate solving。
```

后续可以把 LOST/tetra3rs 作为 star matching adapter 接入。

## 3. 输出内容

单张图 demo 会输出：

```text
*_star_mask.png
*_target_mask.png
*_stars.json
*_targets.json
*_proxy_features.json
```

其中：

```text
star_mask: 星点候选区域。
target_mask: 点状目标和条纹目标候选区域。
stars.json: 星点候选列表，含 x/y/snr/flux/fwhm。
targets.json: 目标候选列表，含 target_type/detector_family/x/y/bbox/snr/confidence。
proxy_features.json: Policy Net 在线可用的下游 proxy features。
```

## 4. 单张图测试命令

在 `STAR_Agent/` 目录执行：

```bash
python scripts/downstream/demo/run_detectors.py \
  --image data/clean/synthetic_v002_targets/images/synthetic_clean_000000.png \
  --output_dir runs/downstream/demo/synthetic_clean_000000
```

如果是服务器真实 clean：

```bash
python scripts/downstream/demo/run_detectors.py \
  --image data/clean/real_selected_v001/images/frame_0120_2018206192942.png \
  --output_dir runs/downstream/demo/real_frame_0120
```

## 5. 为 real selected clean 生成 pseudo masks

真实 clean 通常没有严格 GT，因此需要伪标签。

命令：

```bash
python scripts/downstream/build_real_clean_pseudo_labels.py \
  --clean_root data/clean/real_selected_v001 \
  --overwrite
```

如果你的真实图像还意外多了一层子目录，比如：

```text
data/clean/real_selected_v001/images/clean_data/*.png
```

可以先移动到 `images/` 下，或者临时加递归：

```bash
python scripts/downstream/build_real_clean_pseudo_labels.py \
  --clean_root data/clean/real_selected_v001 \
  --recursive \
  --overwrite
```

输出：

```text
data/clean/real_selected_v001/masks/star_pseudo/
data/clean/real_selected_v001/masks/target_pseudo/
data/clean/real_selected_v001/masks/background_pseudo/
data/clean/real_selected_v001/masks/valid_pseudo/
data/clean/real_selected_v001/labels/stars_pseudo/
data/clean/real_selected_v001/labels/targets_pseudo/
data/clean/real_selected_v001/labels/downstream_proxy/
data/clean/real_selected_v001/manifest_pseudo.jsonl
```

注意：这些是 pseudo label，不是严格 GT。manifest 中会写：

```text
mask_source: pseudo_detector_v001
mask_confidence: 0.7
```

## 6. Policy Net 当前可用 proxy features

当前 demo 和 pseudo label 脚本会输出：

```text
detected_star_count_norm
star_snr_mean
star_fwhm_mean
target_candidate_count
target_candidate_snr
target_confidence
point_blob_count
short_streak_count
```

预留但当前未接 plate solver 的字段：

```text
plate_solve_success
solver_confidence
matched_star_count_norm
unmatched_candidate_ratio
reprojection_error_norm
```

这些后续由 LOST / tetra3rs / astrometry adapter 填充。

## 7. 后续接入顺序

推荐顺序：

```text
1. 用当前轻量 detector 给 real selected clean 生成 pseudo labels。
2. 抽查 real pseudo masks 是否合理，必要时调整 threshold。
3. 用 synthetic GT mask 评估 detector 质量，得到 detector recall / false alarm。
4. 将 downstream proxy features 接入 Star-DepictQA-Lite risk head。
5. 将 downstream proxy features 接入 Policy Net state。
6. 后续再补 LOST / tetra3rs 适配器，提供 plate solving 相关 proxy。
```

## 8. 外部工具正式接入

当前已补充两个外部工具入口：

```text
scripts/downstream/run_tetra3rs.py
scripts/downstream/build_real_clean_tetra3rs_masks.py
scripts/downstream/run_astride.py
```

其中 `tetra3rs` 用于真实 clean-data 的 star pseudo mask，`ASTRiDE` 用于 short-streak target mask。详细命令见：

```text
docs/downstream_external_tools.md
```
