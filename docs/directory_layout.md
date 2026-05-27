# STAR-Agent 目录结构说明

本文档说明当前 STAR-Agent 中各类核心模块应该放在哪里。

## 1. Executor

Executor 是真正执行 restoration 的工具，放在：

```text
star_agent/executors/
```

子目录：

```text
star_agent/executors/denoising/
star_agent/executors/desmear/
star_agent/executors/decosmic_ray/
star_agent/executors/dead_pixel_repair/
star_agent/executors/destray_light/
star_agent/executors/motion_deblurring/
```

其中：

```text
denoising: noise / background noise
desmear: smear / row-column readout artifact
decosmic_ray: cosmic ray
dead_pixel_repair: dead pixels / hot pixels / dead columns
destray_light: dqg / solar stray light
motion_deblurring: motion blur
```

重模型 teacher 或 baseline 放在：

```text
star_agent/executors/teachers/
```

量化、蒸馏、评估相关逻辑分别放在：

```text
star_agent/executors/quantization/
star_agent/executors/distillation/
star_agent/executors/evaluation/
```

Executor 配置放在：

```text
configs/executors/
```

Executor 训练、测试、导出脚本放在：

```text
scripts/executors/
```

Executor 权重和导出模型目录：

```text
checkpoints/executors/
exports/executors/
```

## 2. 下游任务与检测器

下游任务模块放在：

```text
star_agent/downstream/
```

星点检测和质心定位：

```text
star_agent/downstream/star_detection/
```

星图匹配 / plate solving 适配器：

```text
star_agent/downstream/star_matching/lost_adapter/
star_agent/downstream/star_matching/tetra3rs_adapter/
star_agent/downstream/star_matching/astrometry_adapter/
```

小目标检测器：

```text
star_agent/downstream/target_detection/log_blob_detector/
star_agent/downstream/target_detection/streak_detector/
star_agent/downstream/target_detection/astride_adapter/
```

在线 proxy 指标和离线 GT 指标：

```text
star_agent/downstream/proxy_metrics/
star_agent/downstream/offline_metrics/
```

下游任务配置和脚本：

```text
configs/downstream/
scripts/downstream/
```

## 3. Star-DepictQA-Lite

Star-DepictQA-Lite 是轻量状态感知模块，放在：

```text
star_agent/star_depictqa_lite/
```

主要子目录：

```text
models/backbones: MobileNet / ShuffleNet / Small CNN backbone
models/heads: degradation head / severity head / risk head / ranking head
features: 图像统计和 prior vector 特征
losses: 多任务损失
datasets: dataset / dataloader
training: 训练逻辑
inference: 推理逻辑
evaluation: 指标评估
export: ONNX / INT8 / FPGA-friendly 导出
```

配置、脚本、权重和导出目录：

```text
configs/star_depictqa_lite/
scripts/star_depictqa_lite/
checkpoints/star_depictqa_lite/
exports/star_depictqa_lite/
```

训练 meta 数据放在：

```text
data/star_depictqa_lite_meta/<domain>/<clean_source_name>/
```

## 4. Policy Net

Policy Net 是轻量规划器，放在：

```text
star_agent/policy/
```

主要子目录：

```text
models: MLP / GRU / Tiny Transformer policy model
features: policy state feature 构建
reward: 离线 reward 计算
q_learning: Q head / Q label 训练
behavior_cloning: LLM / oracle 行为克隆
rollout: 离线 rollout 样本生成
memory: verified / unverified memory 读写
retrieval: 结构化经验检索特征
safety: stop / rollback / damage guard
training: 训练逻辑
inference: 在线推理
evaluation: policy 指标评估
```

配置、脚本、权重和导出目录：

```text
configs/policy/
scripts/policy/
checkpoints/policy/
exports/policy/
```

经验库和离线 rollout 数据放在：

```text
data/policy_data/<domain>/<clean_source_name>/
```

## 5. Runtime

完整 agent 在线运行时的状态、调度、回滚和日志逻辑放在：

```text
star_agent/runtime/
```

子目录：

```text
state: 当前图像状态和 episode state
planner: policy 输出到动作选择的逻辑
scheduler: executor 调用调度
rollback: 回滚和候选缓存
logging: 在线执行日志
```

运行入口脚本：

```text
scripts/runtime/
```

运行日志：

```text
runs/runtime/
```

## 6. data 与 runs 的区别

```text
data/: 正式数据资产，会被训练、测试、评估读取。
runs/: 某次运行的日志、命令记录、临时输出和进度记录。
```

例如：

```text
data/clean/synthetic_v002_targets/
```

是生成好的 clean 数据集。

```text
runs/data_generation/clean_simulation/build_clean_2000.log
```

是这一次生成 clean 数据集的运行日志。

## 7. Real / Synthetic 数据必须分开

当前 `data/clean/` 下会同时存在真实筛选 clean 和星表仿真 clean，例如：

```text
data/clean/real_selected_v001/
data/clean/synthetic_v002_targets/
```

这两类 clean 的用途不同：

```text
synthetic_v002_targets: 有完整星点 mask、目标 mask、目标 label，适合生成退化、训练 executor、训练 Star-DepictQA-Lite、构建离线 policy reward。
real_selected_v001: 更接近真实观测风格，但通常缺少严格 GT，适合真实域验证、视觉校准和 domain gap 分析。
```

因此从 clean 派生出来的所有数据都必须带上两层来源标识：

```text
<domain>/<clean_source_name>
```

其中 `domain` 当前取值：

```text
synthetic
real
```

推荐目录结构如下：

```text
data/degraded/synthetic/synthetic_v002_targets/single/
data/degraded/synthetic/synthetic_v002_targets/double/
data/degraded/synthetic/synthetic_v002_targets/triple/

data/degraded/real/real_selected_v001/single/
data/degraded/real/real_selected_v001/double/
data/degraded/real/real_selected_v001/triple/
```

Executor 训练/测试数据同样分开：

```text
data/tool_datasets/synthetic/synthetic_v002_targets/denoising/
data/tool_datasets/synthetic/synthetic_v002_targets/destray_light/
data/tool_datasets/synthetic/synthetic_v002_targets/motion_deblurring/

data/tool_datasets/real/real_selected_v001/denoising/
data/tool_datasets/real/real_selected_v001/destray_light/
data/tool_datasets/real/real_selected_v001/motion_deblurring/
```

Star-DepictQA-Lite 和 Policy Net 相关数据也按相同规则存放：

```text
data/star_depictqa_lite_meta/synthetic/synthetic_v002_targets/
data/star_depictqa_lite_meta/real/real_selected_v001/

data/policy_data/synthetic/synthetic_v002_targets/
data/policy_data/real/real_selected_v001/
```

清洁数据源的统一登记文件为：

```text
data/domain_registry.yaml
```

后续脚本应优先读取 `domain_registry.yaml` 或配置文件中的 `clean_source` 字段。不要再把新数据直接写到旧的全局路径：

```text
data/degraded/single/
data/degraded/double/
data/degraded/triple/
data/tool_datasets/denoising/
```

这些旧路径已经废弃并删除。后续 executor 数据只允许写入带 domain/source 的新路径，避免 synthetic、real-selected 和 true-real-degraded 数据混在一起。
