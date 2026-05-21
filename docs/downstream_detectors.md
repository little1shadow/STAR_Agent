# STAR-Agent 下游检测器补充方案

本文档对应 `docs/star_agent_construct.md` 第 8 节“下游任务反馈”。

## 1. 为什么现在要先补下游检测器

如果只生成 synthetic clean 和 degradation，还只能训练 restoration 本身。STAR-Agent 的论文核心不是单纯追求 PSNR，而是强调：

```text
restoration 是否保护了星点、目标和下游任务能力。
```

因此需要先补一套下游感知工具。当前真实 clean 阶段先只产生：

```text
star_mask / background_mask / valid_mask
star centroid labels
real clean star/background pseudo labels
```

target 和 online downstream proxy 暂时不在真实 clean 阶段生成，因为真实 clean 中没有可靠目标 GT。目标会像 synthetic clean 一样在后续步骤单独注入。生成后的 star/background 信息后续会进入：

```text
1. 真实 clean 的伪标签记录
2. 后续真实域 degradation 的背景/星点区域约束
3. executor 的区域加权 loss
4. Star-DepictQA-Lite 的 risk label / risk head
5. 后续 target 注入后的下游任务评价
```

## 2. 当前第一阶段补哪些 detector

根据 construct 文档，第一阶段先补两个下游任务：

```text
1. 星点检测 / 质心定位
2. 小目标检测
```

当前已经补入的工具分两层：

```text
star_detection/blob/detector.py
  轻量备用星点 blob detector，用于 tetra3rs 不可用时生成 star/background pseudo mask。

target_detection/log_blob_detector/detector.py
  点状弱目标 detector，后续 target 注入和下游检测时使用。

target_detection/streak_detector/detector.py
  短条纹目标 detector，后续 target 注入和下游检测时使用。

star_matching/tetra3rs_adapter/
  正式真实 clean star pseudo mask 生成工具，优先使用。

target_detection/astride_adapter/
  ASTRiDE 条纹目标检测工具，后续 short-streak target 阶段使用。
```

真实 clean 阶段不使用 target detector 的原因：

```text
1. 真实 clean 中没有可靠 target GT。
2. 真实 clean 可能存在未知弱目标，直接检测会把未知内容误标成监督标签。
3. target 会在后续步骤像 synthetic clean 一样单独注入，届时会有明确 target mask/label。
```

LOST 后续作为 star matching / plate solving 对照工具接入。

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

如果只是临时看真实图的检测效果，可以跑 demo；但它不会作为真实 clean 数据集的正式标签来源：

```bash
python scripts/downstream/demo/run_detectors.py \
  --image data/clean/real_selected_v001/images/frame_0120_2018206192942.png \
  --output_dir runs/downstream/demo/real_frame_0120
```

## 5. 为 real selected clean 生成 star/background pseudo masks

真实 clean 通常没有严格 GT，因此需要星点/背景伪标签。优先使用 tetra3rs：

```bash
conda run -n star_downstream python scripts/downstream/build_real_clean_tetra3rs_masks.py \
  --clean_root data/clean/real_selected_v001 \
  --recursive \
  --overwrite
```

输出：

```text
data/clean/real_selected_v001/masks/star_pseudo_tetra3rs/
data/clean/real_selected_v001/masks/background_pseudo_tetra3rs/
data/clean/real_selected_v001/masks/valid_pseudo_tetra3rs/
data/clean/real_selected_v001/labels/stars_tetra3rs/
data/clean/real_selected_v001/labels/tetra3rs_metrics/
data/clean/real_selected_v001/manifest_tetra3rs_pseudo.jsonl
```

如果 tetra3rs 不可用，可以使用轻量 detector 备用：

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
data/clean/real_selected_v001/masks/background_pseudo/
data/clean/real_selected_v001/masks/valid_pseudo/
data/clean/real_selected_v001/labels/stars_pseudo/
data/clean/real_selected_v001/labels/star_pseudo_metrics/
data/clean/real_selected_v001/manifest_pseudo.jsonl
```

注意：这些是 pseudo label，不是严格 GT。manifest 中会写：

```text
mask_source: pseudo_detector_v001
mask_confidence: 0.65
has_target: false
target_policy: not_injected_yet
```

## 6. Policy Net 后续可用 proxy features

真实 clean 的 star/background mask 生成脚本当前不输出 proxy。等 target 注入、executor 恢复和下游评价流程接上后，再统一生成：

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
1. 用 tetra3rs 给 real selected clean 生成 star/background pseudo labels。
2. 抽查 real pseudo masks 是否合理，必要时调整 threshold。
3. 后续像 synthetic clean 一样给 real clean 注入 target，生成明确 target mask/label。
4. 再生成 degradation，并让 target mask / star mask / bg mask 都能溯源。
5. executor 恢复之后，再计算 downstream proxy features。
6. 后续再补 LOST / plate solving 指标，提供正式 star matching proxy。
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
