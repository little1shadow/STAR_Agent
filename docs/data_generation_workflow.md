# STAR-Agent 数据生成流程指南

本文档记录当前 STAR-Agent 中 clean data、single degradation、double degradation、triple degradation 的推荐生成顺序和命令。

## 1. 数据域必须分开

当前 clean data 分为两类：

```text
data/clean/synthetic_v002_targets/
data/clean/real_selected_v001/
```

后续所有 degraded data 必须继续按数据来源分开：

```text
data/degraded/synthetic/synthetic_v002_targets/
data/degraded/real/real_selected_v001/
```

不要再把新数据写入旧路径：

```text
data/degraded/single/
data/degraded/double/
data/degraded/triple/
```

## 2. Single Degradation 输出内容

每一种 single degradation 的目录结构如下：

```text
data/degraded/<domain>/<clean_source_name>/single/<degradation>/<mode>/level_<1-5>/
├── images/
├── masks/
├── fields/
├── meta/
└── lineage/
```

各目录含义：

```text
images/: 带退化的 LQ 图像。
masks/: 当前退化影响区域的二值 mask。
fields/: 当前退化的连续强度场，例如杂散光场、噪声差异场、运动模糊差异场。
meta/: 当前样本的参数、路径、统计量和 clean 来源信息。
lineage/: 当前样本的显式溯源链，后续 double/triple 和工具训练集优先读取这里。
```

每张 degraded image 至少可以通过 `meta/*.json` 和 `lineage/*.json` 追溯到：

```text
clean image path
clean image id
clean source domain/name
clean star mask
clean target mask
clean background mask
clean valid mask
clean stars label
clean targets label
degradation type
degradation mode
degradation level
random seed
degradation mask
degradation field
```

这样后续构建 executor 训练集时，可以直接知道：

```text
input: 当前带退化图像
target: 对应 clean 图像，或多退化父图
mask: 当前工具应该关注的退化区域
lineage: 该图从哪个 clean/parent 生成，经过了哪些退化步骤
```

## 3. 生成 Synthetic Single Degradation

推荐先在 `STAR_Agent/` 目录下执行。

完整生成所有 single degradation：

```bash
python scripts/data_generation/build_single_degradation.py \
  --config configs/data_generation/degradation_single.yaml
```

输出位置：

```text
data/degraded/synthetic/synthetic_v002_targets/single/
```

默认规则：

```text
每个 degradation/mode/level 补到 200 张。
level 已经 >= 200 张则跳过。
每生成 10 张重新统计一次当前 level 数量。
每个 level 写入时有 lock，支持多终端并行。
```

## 4. 多终端并行生成 Synthetic Single

建议一个终端生成一类 degradation：

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation noise
```

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation smear
```

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation cosmic_ray
```

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation dead_pixels
```

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation dqg
```

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation solar_stray_light
```

```bash
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single.yaml --degradation motion_blur
```

如果只想生成某个 mode 或某个 level：

```bash
python scripts/data_generation/build_single_degradation.py \
  --config configs/data_generation/degradation_single.yaml \
  --degradation noise \
  --mode gaussian \
  --level 1
```

## 5. 小规模测试命令

正式生成前建议先小规模测试：

```bash
python scripts/data_generation/build_single_degradation.py \
  --config configs/data_generation/degradation_single.yaml \
  --degradation noise \
  --mode gaussian \
  --level 1 \
  --per_level_target 5
```

只查看计划，不生成图片：

```bash
python scripts/data_generation/build_single_degradation.py \
  --config configs/data_generation/degradation_single.yaml \
  --degradation noise \
  --mode gaussian \
  --level 1 \
  --dry_run
```

## 6. 生成 Real Selected Single Degradation

等 synthetic single 完成后，再生成 real selected single：

```bash
python scripts/data_generation/build_single_degradation.py \
  --config configs/data_generation/degradation_single_real_selected.yaml
```

输出位置：

```text
data/degraded/real/real_selected_v001/single/
```

注意：真实 clean 通常缺少严格星点/目标 GT，因此更适合真实域验证、视觉校准和 domain gap 分析，不建议作为第一轮 executor 监督训练主数据。

## 7. Double Degradation 生成原则

Double degradation 不能简单从 clean 重新随机叠加两种退化。

必须满足严格溯源：

```text
clean A
├── single AS = A + smear
├── single AD = A + dqg
└── double ASD = AS + 同参数 dqg，或 AD + 同参数 smear
```

也就是说，double 图必须能够在 `lineage` 中找到：

```text
clean parent
first single parent
second single support
每一步 degradation 参数
每一步 mask/field
```

计划命令形式：

```bash
python scripts/data_generation/build_multi_degradation.py \
  --config configs/data_generation/degradation_multi.yaml \
  --stage double
```

该脚本需要在 single 全部完成后再运行。

## 8. Triple Degradation 生成原则

Triple degradation 必须基于 double parent 继续生成：

```text
clean A
single: AS, AD, AM
double: ASD
triple: ASDM = ASD + 与 AM 对应的 motion_blur 参数
```

triple 的 lineage 必须包含：

```text
clean parent
single support parents
double parent
third degradation step
完整 step list
```

计划命令形式：

```bash
python scripts/data_generation/build_multi_degradation.py \
  --config configs/data_generation/degradation_multi.yaml \
  --stage triple
```

## 9. 推荐总体顺序

```text
1. synthetic clean with targets
2. synthetic single degradation
3. synthetic double degradation
4. synthetic triple degradation
5. synthetic tool datasets
6. synthetic star_depictqa_lite meta
7. synthetic policy rollout data
8. real selected single degradation
9. real selected validation / domain gap analysis
```

当前已经可执行的是 synthetic/real single degradation。Double/triple 需要严格 lineage 生成器补齐后再跑。
