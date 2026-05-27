# Executor Tool Dataset Workflow

本文档说明 STAR-Agent 中 executor 工具训练/测试数据如何生成、如何分域组织，以及后续训练时如何组合 synthetic 和 real-selected 数据。

## 1. 数据域

Executor 数据不再混在一个全局目录中，而是按 domain/source 分开：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/
data/tool_datasets/real/real_selected_v001/
data/tool_datasets/real_degraded/unpaired_v001/
```

含义：

```text
synthetic/synthetic_v002_targets:
  完全仿真 clean + 仿真目标 + 仿真 degradation。
  GT 和 mask 最可靠，作为 executor Stage 1 主训练数据。

real/real_selected_v001:
  真实 clean 背景 + 注入目标 + 仿真 degradation。
  有 paired target，但 star/background mask 多数是伪标签。
  适合 Stage 2/3 域适配和真实背景测试。

real_degraded/unpaired_v001:
  真实退化图，无 paired clean GT。
  只用于无参考评价、下游任务评价和人工视觉检查。
```

## 2. 目录结构

每个 domain/source 下按 subtask 和 tool 分层：

```text
data/tool_datasets/<domain>/<source>/<subtask>/<tool_id>/
  train/
    input/
    target/
    degradation_mask/
    degradation_field/
    star_mask/
    target_mask/
    background_mask/
    valid_mask/
    meta/
    manifest.jsonl
  val/
  test/
  dataset_summary.json
```

其中：

```text
input:
  当前带退化图。

target:
  只移除当前 executor 负责退化后的目标图。
  对多退化样本，target 不是 clean，而是保留其他退化的父图/重放图。

degradation_mask:
  当前 executor 负责移除的退化区域 mask。

star_mask / target_mask / background_mask / valid_mask:
  从 clean manifest 继承，用于 task-aware loss 和下游指标。

manifest.jsonl:
  每一行记录 input、target、mask、domain、clean_image_id、stage、removed_degradations 等信息。
```

## 3. Subtask 到 Degradation 映射

```text
denoising          <- noise
desmear            <- smear
decosmic_ray       <- cosmic_ray
dead_pixel_repair  <- dead_pixels
destray_light      <- dqg, solar_stray_light
motion_deblurring  <- motion_blur
```

## 4. 多退化 Target 规则

核心规则：

```text
训练某个 executor 时，只移除它负责的退化，保留其他退化。
```

例子：

```text
input  = noise + dqg
denoising target = dqg

action = destray_light
input  = noise + dqg
destray_light target = noise

input = cosmic_ray + noise + dqg
decosmic_ray target = noise + dqg
```

脚本会优先复用已有 clean、parent、support single 图作为 target。只有当需要“移除中间一步退化并保留前后退化”时，才会按 lineage 中保存的 mode/level/seed 重放生成 target。

## 5. 生成 Synthetic Executor 数据

在服务器 `STAR_Agent/` 根目录执行：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml
```

只生成某一个 subtask：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask denoising
```

只生成某个工具版本：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask denoising \
  --tool swinir_v001
```

先检查数量但不写数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --dry_run
```

## 6. 生成 Real-Selected Executor 数据

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml
```

只生成 destray_light 的 real-selected 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask destray_light
```

## 7. 文件落盘策略

默认：

```text
link_mode: hardlink
```

优点是基本不额外占空间。如果 hardlink 失败，脚本会自动 copy fallback。

如果要把 tool dataset 单独打包搬走，可以改成：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --link_mode copy
```

## 8. 训练时如何组合数据

不要把 synthetic 和 real-selected 物理合并成一个不可追踪目录。推荐训练脚本读取两套 manifest，用 sampler 控制比例。

推荐三阶段：

```text
Stage 1: synthetic pretrain
  synthetic: 100%
  real-selected: 0%

Stage 2: domain fine-tune
  synthetic: 60% - 70%
  real-selected: 30% - 40%

Stage 3: real-biased task-aware fine-tune
  synthetic: 30% - 50%
  real-selected: 50% - 70%
```

Stage mix 建议：

```text
早期:
  single : double : triple = 6 : 3 : 1

中后期:
  single : double : triple = 4 : 4 : 2
```

## 9. 测试结果必须分域汇报

每个 executor 至少单独汇报：

```text
synthetic_test:
  看理想 GT 条件下的上限。

real_selected_test:
  看真实背景适应能力。

true_real_degraded_eval:
  没有 clean GT 时，看无参考指标、下游检测指标和人工视觉质量。
```

不要只给混合测试集平均值，否则无法判断模型到底是 restoration 能力不足，还是 domain gap 造成的问题。

## 10. True Real Degraded Unpaired Eval

真实退化图通常没有 paired clean，因此不适合用 `build_tool_datasets.py` 生成监督训练集。对应配置是：

```text
configs/data_generation/tool_dataset_real_degraded_eval.yaml
```

推荐目录：

```text
data/tool_datasets/real_degraded/unpaired_v001/denoising_eval/
data/tool_datasets/real_degraded/unpaired_v001/desmear_eval/
data/tool_datasets/real_degraded/unpaired_v001/decosmic_ray_eval/
data/tool_datasets/real_degraded/unpaired_v001/dead_pixel_repair_eval/
data/tool_datasets/real_degraded/unpaired_v001/destray_light_eval/
data/tool_datasets/real_degraded/unpaired_v001/motion_deblurring_eval/
```

这部分后续只做：

```text
1. 无参考图像质量统计
2. 下游星点/目标检测 proxy
3. 人工视觉检查
4. 与 synthetic_test / real_selected_test 分开汇报
```
