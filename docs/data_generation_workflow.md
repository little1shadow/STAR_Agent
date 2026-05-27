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

单独生成每一类退化
python scripts/data_generation/build_single_degradation.py --config configs/data_generation/degradation_single_real_selected.yaml --degradation noise --per_level_target 50
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

```bash
python scripts/data_generation/build_multi_degradation.py \
  --config configs/data_generation/degradation_multi_real_selected.yaml \
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

```bash
python scripts/data_generation/build_multi_degradation.py \
  --config configs/data_generation/degradation_multi_real_selected.yaml \
  --stage triple
```
## 9. 推荐总体顺序

```text
1. synthetic clean with targets
2. synthetic single degradation
3. synthetic double degradation
4. synthetic triple degradation
5. synthetic executor/tool datasets
6. synthetic star_depictqa_lite meta
7. synthetic policy rollout data
8. real selected clean masks / targets
9. real selected single degradation
10. real selected double degradation
11. real selected triple degradation
12. real selected executor/tool datasets
13. true real degraded unpaired evaluation data
```

当前 single/double/triple degradation 和 executor/tool dataset 都按 domain/source 分层管理。生成 executor 数据前，需要确认对应 domain 的 single、double、triple 已经生成完成。

## 10. Executor / Tool Dataset 生成总原则

Executor 数据不再写入旧的全局目录，例如：

```text
data/tool_datasets/denoising/
data/tool_datasets/destray_light/
data/tool_datasets/motion_deblurring/
```

这些旧目录已经删除。新数据必须按 domain/source/subtask/tool 分开：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/<subtask>/<tool_id>/
data/tool_datasets/real/real_selected_v001/<subtask>/<tool_id>/
data/tool_datasets/real_degraded/unpaired_v001/<subtask>_eval/
```

三类数据含义：

```text
synthetic/synthetic_v002_targets:
  完全仿真 clean + 仿真目标 + 仿真 degradation。
  mask 和 GT 最可靠，优先用于深度 executor 的 Stage 1 训练。

real/real_selected_v001:
  真实 clean 背景 + 注入目标 + 仿真 degradation。
  有 paired target，但 star/background mask 多为伪标签。
  适合 Stage 2/3 域适配和真实背景测试。

real_degraded/unpaired_v001:
  真实退化图，没有 paired clean GT。
  只用于无参考评价、下游检测评价和人工视觉检查。
```

生成前必须已经完成：

```text
1. 对应 domain 的 single degradation
2. 对应 domain 的 double degradation
3. 对应 domain 的 triple degradation
```

否则 tool dataset 会扫描不到样本。

## 11. Executor Dataset 输出结构

每个 subtask/tool_id 下会生成：

```text
data/tool_datasets/<domain>/<source>/<subtask>/<tool_id>/
├── train/
│   ├── input/
│   ├── target/
│   ├── degradation_mask/
│   ├── degradation_field/
│   ├── star_mask/
│   ├── target_mask/
│   ├── background_mask/
│   ├── valid_mask/
│   ├── meta/
│   └── manifest.jsonl
├── val/
├── test/
└── dataset_summary.json
```

字段含义：

```text
input:
  当前带退化图像。

target:
  当前 executor 训练目标。
  对 single degradation，通常是 clean。
  对 double/triple degradation，不一定是 clean，而是只移除当前 executor 负责的退化后得到的父图/重放图。

degradation_mask:
  当前 executor 负责修复的退化区域 mask。

degradation_field:
  当前 executor 负责退化的连续强度场，可用于可解释分析或 field-based loss。

star_mask / target_mask / background_mask / valid_mask:
  从 clean 数据继承，用于 task-aware loss 和下游指标。

manifest.jsonl:
  每一行对应一个 input-target pair，记录 domain、clean_image_id、stage、source path、removed_degradations 等信息。
```

注意：划分 train/val/test 时按 `clean_image_id` 分，而不是按文件随机分。这样可以避免同一张 clean 图的不同退化版本同时进入训练集和测试集。

## 12. 多退化样本 Target 规则

核心原则：

```text
训练某个 executor 时，只移除该 executor 负责的退化，保留其他退化。
```

例如：

```text
input = noise + dqg
训练 denoising:
  target = dqg

input = noise + dqg
训练 destray_light:
  target = noise

input = cosmic_ray + noise + dqg
训练 decosmic_ray:
  target = noise + dqg

input = motion_blur + solar_stray_light + noise
训练 motion_deblurring:
  target = solar_stray_light + noise
```

这样做的目的：

```text
1. single degradation 教模型“应该去掉什么”。
2. double/triple degradation 教模型“不要误删其他退化、星点和目标”。
3. 每个 executor 保持职责清晰，避免退化成 all-in-one restoration model。
```

脚本会优先复用已有 clean、parent、support single 图作为 target。只有在无法直接复用时，才会按 lineage 中保存的 mode/level/seed 重放生成 target。

## 13. 一次性生成 Synthetic Executor 数据

在服务器 `STAR_Agent/` 根目录执行：

```bash
cd ~/job_example/STAR_Agent

python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml
```

输出：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/
```

该命令会生成配置中所有 subtask 和 tool：

```text
denoising:
  tiny_denoise_v001
  swinir_v001
  mprnet_v001

desmear:
  tiny_desmear_v001

decosmic_ray:
  astroscrappy_eval_v001
  tiny_decosmic_v001

dead_pixel_repair:
  classical_eval_v001
  tiny_deadpixel_v001

destray_light:
  tiny_destray_v001
  dehazeformer_v001

motion_deblurring:
  tiny_deblur_v001
  restormer_v001
  mprnet_v001
```

如果数据量很大，建议先用 `--dry_run` 看数量：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --dry_run
```

## 14. 一次性生成 Real-Selected Executor 数据

```bash
cd ~/job_example/STAR_Agent

python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml
```

输出：

```text
data/tool_datasets/real/real_selected_v001/
```

先检查数量但不写文件：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --dry_run
```

注意：real-selected 有 paired target，但 star/background mask 置信度低于 synthetic。训练时不要把 real-selected 和 synthetic 物理合并成一个目录，而是训练脚本分别读取 manifest，用 sampler 控制比例。

## 15. 分别生成深度学习 Executor 数据

### 15.1 Denoising

负责退化：

```text
noise
```

生成 synthetic 的 SwinIR / MPRNet / tiny denoise 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask denoising
```

只生成 SwinIR：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask denoising \
  --tool swinir_v001
```

只生成 MPRNet denoising：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask denoising \
  --tool mprnet_v001
```

生成 real-selected denoising 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask denoising
```

输出示例：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/denoising/swinir_v001/
data/tool_datasets/synthetic/synthetic_v002_targets/denoising/mprnet_v001/
data/tool_datasets/real/real_selected_v001/denoising/swinir_v001/
```

### 15.2 Destray Light

负责退化：

```text
dqg
solar_stray_light
```

生成 synthetic 的 DehazeFormer / tiny destray 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask destray_light
```

只生成 DehazeFormer：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask destray_light \
  --tool dehazeformer_v001
```

生成 real-selected destray_light 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask destray_light
```

输出示例：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/destray_light/dehazeformer_v001/
data/tool_datasets/real/real_selected_v001/destray_light/dehazeformer_v001/
```

### 15.3 Motion Deblurring

负责退化：

```text
motion_blur
```

生成 synthetic 的 Restormer / MPRNet / tiny deblur 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask motion_deblurring
```

只生成 Restormer：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask motion_deblurring \
  --tool restormer_v001
```

只生成 MPRNet deblurring：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask motion_deblurring \
  --tool mprnet_v001
```

生成 real-selected motion_deblurring 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask motion_deblurring
```

输出示例：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/motion_deblurring/restormer_v001/
data/tool_datasets/synthetic/synthetic_v002_targets/motion_deblurring/mprnet_v001/
data/tool_datasets/real/real_selected_v001/motion_deblurring/restormer_v001/
```

### 15.4 Desmear

负责退化：

```text
smear
```

生成 synthetic desmear 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask desmear
```

生成 real-selected desmear 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask desmear
```

输出示例：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/desmear/tiny_desmear_v001/
data/tool_datasets/real/real_selected_v001/desmear/tiny_desmear_v001/
```

## 16. 生成非深度方法的测试/评价数据

非深度方法一般不需要训练，但仍然需要统一格式的测试数据，方便计算 PSNR/SSIM、无参考指标和下游指标。

当前非深度方法主要包括：

```text
decosmic_ray:
  astroscrappy_eval_v001
  用于 L.A.Cosmic / Astro-SCRAPPY 类 cosmic ray 修复评价。

dead_pixel_repair:
  classical_eval_v001
  用于坏点 mask + interpolation / inpaint 类方法评价。
```

### 16.1 Astro-SCRAPPY / L.A.Cosmic Cosmic Ray 测试数据

Synthetic 测试数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask decosmic_ray \
  --tool astroscrappy_eval_v001
```

Real-selected 测试数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask decosmic_ray \
  --tool astroscrappy_eval_v001
```

输出：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/decosmic_ray/astroscrappy_eval_v001/test/
data/tool_datasets/real/real_selected_v001/decosmic_ray/astroscrappy_eval_v001/test/
```

配置中该工具的 split 是：

```text
train: 0.0
val: 0.0
test: 1.0
```

因此它只生成测试/评价集，不生成训练集。

### 16.2 Dead Pixel Classical Repair 测试数据

Synthetic 测试数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask dead_pixel_repair \
  --tool classical_eval_v001
```

Real-selected 测试数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset_real_selected.yaml \
  --subtask dead_pixel_repair \
  --tool classical_eval_v001
```

输出：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/dead_pixel_repair/classical_eval_v001/test/
data/tool_datasets/real/real_selected_v001/dead_pixel_repair/classical_eval_v001/test/
```

配置中该工具同样只生成测试/评价集：

```text
train: 0.0
val: 0.0
test: 1.0
```

### 16.3 可选 Tiny Decosmic / Tiny Dead Pixel 数据

如果后续决定训练轻量深度模型，可以生成 optional tiny 数据：

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask decosmic_ray \
  --tool tiny_decosmic_v001
```

```bash
python scripts/data_generation/build_tool_datasets.py \
  --config configs/data_generation/tool_dataset.yaml \
  --subtask dead_pixel_repair \
  --tool tiny_deadpixel_v001
```

这两个是可选训练候选，不是第一阶段必须项。

## 17. True Real Degraded 无配对评价数据

真实退化图通常没有 clean GT，因此不能用 `build_tool_datasets.py` 生成 paired train/val/test。

对应配置：

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

## 18. 训练时如何组合 Synthetic 和 Real-Selected

不要把两类数据物理合并成一个目录。推荐训练脚本分别读取两套 manifest：

```text
synthetic manifest:
  data/tool_datasets/synthetic/synthetic_v002_targets/<subtask>/<tool_id>/<split>/manifest.jsonl

real-selected manifest:
  data/tool_datasets/real/real_selected_v001/<subtask>/<tool_id>/<split>/manifest.jsonl
```

推荐训练阶段：

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

single/double/triple 比例建议：

```text
早期:
  single : double : triple = 6 : 3 : 1

中后期:
  single : double : triple = 4 : 4 : 2
```

注意：测试结果必须分域汇报，不要只报混合平均值：

```text
synthetic_test
real_selected_test
true_real_degraded_unpaired_eval
```

## 19. 常用检查命令

查看某个工具生成了多少 input/target：

```bash
find data/tool_datasets/synthetic/synthetic_v002_targets/denoising/swinir_v001/train/input \
  -maxdepth 1 -type f -name '*.png' | wc -l

find data/tool_datasets/synthetic/synthetic_v002_targets/denoising/swinir_v001/train/target \
  -maxdepth 1 -type f -name '*.png' | wc -l
```

查看 manifest 前几行：

```bash
head -n 3 data/tool_datasets/synthetic/synthetic_v002_targets/denoising/swinir_v001/train/manifest.jsonl
```

查看某个工具的 summary：

```bash
cat data/tool_datasets/synthetic/synthetic_v002_targets/denoising/swinir_v001/dataset_summary.json
```

查看全部构建 summary：

```bash
cat data/tool_datasets/synthetic/synthetic_v002_targets/_manifests/tool_dataset_summary.json
```

如果本地只同步了目录骨架，没有同步实际 degraded 图片，那么 `--dry_run` 显示 `0 samples` 是正常的；需要在服务器数据完整的位置执行。
