# STAR-Agent 构建方案

`STAR-Agent` 建议全称为：

```text
Space Task-aware Agent for Restoration
```

中文定位：

```text
面向空间图像下游任务的轻量化、任务感知、多退化 restoration agent。
```

它不是简单把自然图像 restoration agent 搬到空间图像上，而是围绕空间图像的三个特点重新设计：

```text
1. 有效目标占空比极低，PSNR/SSIM 容易被大面积背景主导。
2. 弱星点、小目标、质心位置对下游任务极其敏感。
3. 星载/天基平台资源受限，在线模型必须轻量、可控、可解释。
```

---

## 1. 根本动机

### 1.1 空间图像 restoration agent 仍然缺失

自然图像领域已经有大量 restoration agent 或 tool-use restoration 系统，用于处理去噪、去雨、去雾、去模糊、超分、JPEG artifact 等多种退化。但在空间图像领域，目前公开工作更多是单一问题的处理，例如：

```text
星图杂散光抑制
天文图像背景梯度去除
宇宙射线修复
坏点修复
星图运动模糊建模与恢复
星点检测和质心定位
```

这些方法通常是分散的、任务单一的，很少有一个统一系统能同时完成：

```text
退化判别
恢复顺序规划
工具选择
恢复结果反思
下游任务反馈
资源约束下的停止决策
```

因此，STAR-Agent 的第一个动机是补足空间图像领域缺少统一 restoration agent 的空白。

### 1.2 空间图像 restoration 工具碎片化

公开可用工具主要集中在少数退化上。

例如：

```text
cosmic ray: astroscrappy / ccdproc.cosmicray_lacosmic
bad pixel: ccdproc.ccdmask / Siril cosmetic correction / mask interpolation
background / stray-light-like gradient: GraXpert / BSC-Net 思路
motion blur: 星图运动模糊论文较多，但可直接复用的开源 restoration executor 较少
smear: 更偏仪器/列方向校正，专用公开深度工具较少
```

这说明单个工具难以覆盖空间图像中所有退化。因此 STAR-Agent 不应被设计成单模型，而应是一个可扩展的工具调度系统。

### 1.3 为什么不直接做 all-in-one restoration model

空间图像看起来比自然图像简单，大面积是背景，星点和目标稀疏。因此一个自然的质疑是：

```text
为什么不直接训练一个万能 all-in-one restoration model？
```

这个方向可以作为 baseline，但不应作为最终系统的唯一方案，原因包括：

```text
1. 多退化组合空间快速膨胀。
   当前退化可能包括 noise、smear、dqg、solar stray light、cosmic ray、dead pixels、motion blur。
   单退化、双退化、三退化再叠加不同 level 和 mode 后，组合数量很大。

2. 不同退化的形成机理和恢复尺度不同。
   cosmic ray / dead pixel 是局部异常点问题；
   noise 是随机扰动；
   dqg / solar stray light 是大尺度低频背景场；
   smear 是行/列方向结构异常；
   motion blur 是 PSF / 运动核逆问题。
   一个 all-in-one model 容易学成平均化策略。

3. 下游任务目标可能和视觉恢复目标冲突。
   PSNR 更高不一定意味着星点质心更准、小目标召回更高。
   过强去噪可能抹掉弱目标，过强去杂散光可能改变背景和星点亮度。

4. 星载部署需要按需计算。
   all-in-one model 对每张图都跑完整网络，而 agent 可以根据退化类型只调用必要工具。

5. agent 更可解释和可控。
   它能输出退化判断、工具顺序、每一步指标变化和停止原因。
```

因此，STAR-Agent 的目标不是简单替代 all-in-one model，而是提供一个更高层的 restoration decision layer。

可以在实验中把 all-in-one model 作为 baseline：

```text
all-in-one 解决固定图像到图像映射；
STAR-Agent 解决任务感知、资源感知、可解释的恢复决策。
```

### 1.4 为什么需要轻量化

空间图像大多来自天基或星载平台，在线部署受限于：

```text
算力
功耗
内存
实时性
可靠性
可解释性
```

因此不能直接照搬自然图像中的大模型 agent。STAR-Agent 采用两阶段思路：

```text
地面离线阶段：
  可以使用 LLM、重型 executor、GT 指标和下游任务模块探索恢复策略。

星载在线阶段：
  部署 Star-DepictQA-Lite、轻量 Policy Net、轻量 executor 和规则/经验模块。
```

也就是说：

```text
LLM 只作为离线 teacher；
Policy Net 是在线轻量 planner；
重型 executor 只作为 baseline 或可选 teacher；
最终部署采用轻量 executor。
```

---

## 2. STAR-Agent 总体模块

本文档将 STAR-Agent 拆成八个模块：

```text
1. 指标体系
2. Degradation 生成
3. 数据生成部分
4. Executor
5. Star-DepictQA-Lite
6. 下游任务反馈
7. Policy Net
8. 调度流程
```

---

## 3. 指标体系

空间图像恢复不能只看 PSNR/SSIM。建议采用三层指标体系。

### 3.1 第一层：传统图像恢复指标

用于和常规 restoration 方法对比。

| 指标 | 中文 | 作用 | 是否作为核心 |
|---|---|---|---|
| `PSNR` | 峰值信噪比 | 衡量像素级误差 | 只作为基础参考 |
| `SSIM` | 结构相似性 | 衡量局部结构相似度 | 只作为基础参考 |
| `MAE` | 平均绝对误差 | 衡量平均像素偏差 | 可选 |
| `LPIPS` | 感知相似度 | 衡量深层感知差异 | 可选，不一定适合星空 |

问题：

```text
星空图像大面积是背景，PSNR/SSIM 很容易被背景区域主导。
模型即使抹掉弱星点或小目标，也可能获得较高 PSNR。
```

### 3.2 第二层：空间图像专属恢复指标

这一层指标用于评价空间任务相关的信息是否被保留。

| 指标 | 中文 | 核心含义 |
|---|---|---|
| `OARS` | 占空比感知恢复分数 | 按目标、星点、退化区域、背景区域加权评价 |
| `Target-aware PSNR` | 目标区域感知 PSNR | 目标区域权重更高的 PSNR |
| `Star-aware PSNR` | 星点区域感知 PSNR | 星点区域权重更高的 PSNR |
| `Star Preservation Rate` | 星点保持率 | 恢复后有多少真实星点仍能被检测/匹配 |
| `Target Preservation Rate` | 目标保持率 | 恢复后目标是否仍被保留 |
| `Weak Target Recall` | 弱目标召回率 | 对低 SNR 目标的检出能力 |
| `False Star Rate` | 伪星点率 | 恢复是否引入伪星点 |
| `False Target Rate` | 伪目标率 | 恢复是否引入伪目标 |
| `Centroid Error` | 质心误差 | 星点/目标中心偏移 |
| `Flux Error` | 能量/亮度误差 | 星点或目标区域总亮度是否被改变 |
| `Background Residual Error` | 背景残留误差 | 背景区域退化残留程度 |
| `Degradation Residual Score` | 退化残留分数 | 指定退化区域是否被正确修复 |

建议核心综合指标：

```text
OARS = α * Q_target + β * Q_star + γ * Q_deg + δ * Q_bg
```

其中：

```text
Q_target: 目标区域质量
Q_star: 星点区域质量
Q_deg: 退化区域修复质量
Q_bg: 背景稳定性
```

更具体地，可以把四个子项定义为归一化后的分数，数值范围统一到 `[0, 1]`，越大表示越好：

```text
Q_target = 0.4 * Target_Preservation
         + 0.3 * Weak_Target_Recall
         + 0.2 * (1 - Target_Centroid_Error_Norm)
         + 0.1 * (1 - Target_Flux_Error_Norm)

Q_star   = 0.4 * Star_Preservation
         + 0.3 * Plate_Solve_Success
         + 0.2 * (1 - Star_Centroid_Error_Norm)
         + 0.1 * (1 - False_Star_Rate)

Q_deg    = 0.5 * (1 - Degradation_Residual_Norm)
         + 0.3 * (1 - Artifact_Area_Ratio)
         + 0.2 * Local_Structure_Consistency

Q_bg     = 0.4 * (1 - Background_Residual_Norm)
         + 0.3 * (1 - Background_Nonuniformity_Norm)
         + 0.3 * (1 - Background_Noise_Norm)
```

这里的 `Norm` 表示用验证集统计量或任务可接受阈值做归一化。例如质心误差可以除以 `1 pixel` 或下游星图匹配允许的最大误差，背景残差可以除以 clean 背景标准差。

这样设计的好处是：

```text
目标相关指标不会被大面积黑背景稀释；
星点相关指标可以直接约束星图匹配能力；
退化残留项保证 restoration 本身确实有效；
背景项保留，但权重较低，避免 PSNR 被空背景主导。
```

推荐权重关系：

```text
α > β > γ > δ
```

原因：

```text
目标区域最重要；
星点区域影响星图匹配和姿态估计；
退化区域决定恢复是否完成；
背景区域面积最大，但不能主导评价。
```

### 3.3 第三层：下游任务指标

第一篇论文建议只接入两个下游任务：

```text
1. 星点检测 / 质心定位 / 星图匹配
2. 小目标检测，点状目标或条纹目标二选一或都保留接口
```

推荐指标：

| 下游任务 | 指标 | 中文 |
|---|---|---|
| 星点检测 | `Star Detection Recall` | 星点检测召回率 |
| 星点检测 | `False Star Rate` | 伪星点率 |
| 质心定位 | `Centroid Error` | 星点质心误差 |
| 星图匹配 | `Plate-solve Success` | 星图匹配成功率 |
| 星图匹配 | `Matched Star Count` | 匹配星数量 |
| 星图匹配 | `Reprojection Error` | 重投影误差 |
| 小目标检测 | `Target Recall` | 目标召回率 |
| 小目标检测 | `Precision` | 查准率 |
| 小目标检测 | `False Alarm Rate` | 虚警率 |
| 小目标检测 | `mAP` | 平均精度 |
| 弱目标检测 | `Weak Target Recall` | 弱目标召回率 |

### 3.4 指标和 loss 的关系

不是所有指标都直接进入 loss。建议区分：

```text
训练 loss：可导 proxy。
验证/测试 metric：完整任务指标。
```

训练时推荐使用：

```text
L_restore: L1 / Charbonnier / SSIM loss
L_region: 目标/星点/退化/背景区域加权 L1
L_bg: 背景残留和平滑约束
L_flux: 星点/目标区域总能量保持
L_heatmap: 可选的星点/目标热图 proxy loss
```

可以具体写成：

```text
L_total = λ_restore * L_restore
        + λ_region  * L_region
        + λ_bg      * L_bg
        + λ_flux    * L_flux
        + λ_heatmap * L_heatmap
```

其中：

```text
L_restore = Charbonnier(restored - target)
```

`L_restore` 负责整体像素恢复，是最基础的重建项。

```text
L_region = w_target * ||M_target * (restored - target)||_1
         + w_star   * ||M_star   * (restored - target)||_1
         + w_deg    * ||M_deg    * (restored - target)||_1
         + w_bg     * ||M_bg     * (restored - target)||_1
```

`L_region` 负责解决星空图像占空比低的问题。因为背景区域远大于星点和目标区域，如果只用全图 L1，模型很容易学成“背景很好、目标被抹掉也没关系”。因此建议 `w_target > w_star > w_deg > w_bg`。

```text
L_bg = ||M_bg * LowFreq(restored - target)||_1
     + TV(M_bg * restored)
```

`L_bg` 用来约束背景残余和背景平滑性，尤其适合杂散光、dqg、背景梯度这类低频退化。

```text
Flux(restored, M) = sum(M * restored)
Flux(target, M)   = sum(M * target)

L_flux = |Flux(restored, M_star) - Flux(target, M_star)|
       + |Flux(restored, M_target) - Flux(target, M_target)|
```

`L_flux` 约束星点和目标的总亮度，避免去噪或去杂散光时把弱星点、弱目标的能量一起削弱。

```text
L_heatmap = BCE(H_restored, H_gt) 或 MSE(H_restored, H_gt)
```

`L_heatmap` 是可选项。如果同时训练轻量星点/目标检测头，可以让恢复结果更服务于下游检测。

最终评价仍报告：

```text
OARS
Star Preservation Rate
Target Recall
False Alarm Rate
Centroid Error
PSNR / SSIM
Runtime / Params / FLOPs
```

---

## 4. Degradation 生成

退化生成要尽量参考公开方法或物理机理，同时保留可控 mask 和参数记录。

### 4.1 杂散光：solar stray light / lunar stray light / dqg

公开参考和可借鉴方法：

```text
1. GraXpert
   开源天文图像背景梯度去除工具，支持传统插值和 AI 方法，可作为 background / gradient removal baseline。
   链接: https://github.com/Steffenhir/GraXpert/

2. BSC-Net
   面向 star image stray light 的背景抑制网络，包含 background suppression 和 foreground retention 两部分。
   论文中提到部分数据和代码公开。
   链接: https://www.mdpi.com/2072-4292/14/19/4852

3. FY-3C/VIRR external stray light simulation
   基于卫星轨道姿态、太阳入射角、光学散射模型和 light tracing 模拟外部太阳杂散光。
   链接: https://www.mdpi.com/2072-4292/13/24/5037

4. TESS scattered/background correction 思路
   TESSCut / MAST 数据可以用于获取真实空间望远镜 cutout，部分背景变化可作为真实数据分析来源。
   链接: https://mast.stsci.edu/tesscut/docs/index.html

5. Moon / Sun / Earth integrated ray-tracing 思路
   公开文献和工程软件通常把太阳、月球、地球反照光看作视场外强光源或扩展面光源，再结合遮光罩、光学散射、PSF 或非序列光线追迹得到探测器上的 stray light 分布。
   这类方法可以作为更严格的机理参考；如果缺少完整光机参数，则采用“视场外强光源 + 散射核 + 真实样本统计校准”的近似模型。

6. LIME / lunar irradiance model 与 off-axis moon source 示例
   LIME 可提供月球辐照度建模参考；光学软件的 off-axis moon source 示例可以借鉴“月球作为视场外扩展强光源”的建模方式。
   链接: https://lime.uva.es/
   链接: https://optics.ansys.com/hc/en-us/articles/43071106491027-How-to-perform-stray-light-analysis
```

STAR-Agent 中建议的仿真方式：

```text
solar stray light:
  推荐采用“off-axis solar source + scattering kernel + empirical calibration”的机理近似。
  具体做法是把太阳建模为视场外、靠近边缘的极强光源，只允许在图像边缘或角落露出很小一角；
  然后通过非均匀散射核生成不完整弧线、局部高亮弧、宽弧带、弧内分层和大尺度光晕；
  最后用真实 solar_stray_light_reference 的亮度分布、弧线宽度、弧线完整度、边缘 glint 尺寸校准参数。
  这比单纯手工画弧线更接近物理过程，也保留了无光机参数情况下的可实现性。

lunar stray light:
  不建议在第一版中单独作为核心退化类别，因为已有数据中 lunar stray light 与 dqg 视觉模式高度重叠。
  如果后续确实需要保留，可采用“off-axis extended lunar source + low-frequency scatter field”的模型。
  与 solar 不同，月球亮度远弱于太阳，因此通常不应出现极强边缘光源角；
  更合理的表现是柔和、低频、片状或方向性背景抬升。
  训练标签上可以先并入 dqg / diffuse stray light，避免类别边界模糊。

dqg:
  当前使用组内已有代码作为主实现。
  dqg 表现为方向性、片状、大尺度背景入侵；
  不强调明显点光源；
  level 4/5 需要明显增强背景非均匀性。
```

输出 mask：

```text
M_deg_straylight: 杂散光/背景异常区域，用于标记被 stray light 或 dqg 明显影响的位置。
M_bg: 背景区域 mask，只在仿真数据或有伪标签的真实数据中保存。这里的 M_bg 不是杂散光本身，而是“非星点、非目标、非明显退化结构”的背景区域，用于训练背景残差和背景平滑约束。
```

注意：`M_star`、`M_target` 不在杂散光生成函数内部强行生成。它们应该来自 clean 数据的星点/目标标注、仿真注入记录，或后续伪标签流程。退化生成模块只负责输出当前退化的 `M_deg` 和必要的参数记录。

### 4.2 Cosmic ray

公开参考：

```text
1. astroscrappy
   Astro-SCRAPPY 是 L.A.Cosmic 的 Python 实现，可作为 cosmic ray 修复 executor。
   ccdproc 文档也引用其作为 cosmicray_lacosmic 的相关实现。

2. ccdproc.cosmicray_lacosmic
   提供 cosmic ray mask 和修复后的图像，返回 mask 可直接用于训练和评价。
   链接: https://ccdproc.readthedocs.io/en/2.5.1/api/ccdproc.cosmicray_lacosmic.html
```

仿真方式：

```text
点状 cosmic ray: 稀疏高亮尖峰
短线状 cosmic ray: 随机方向短线
blob cosmic ray: 小范围团状亮斑
mixed: 点、短线、团状混合
```

level 控制：

```text
数量
强度
长度
宽度
是否成簇
```

输出 mask：

```text
M_cosmic: cosmic ray 像素区域
```

### 4.3 Dead pixel / hot pixel

公开参考：

```text
1. ccdproc.ccdmask
   用于构建 bad pixel mask。
   链接: https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/08-02-Creating-a-mask.html

2. Siril cosmetic correction
   天文图像 hot/cold pixel 修复工具，可参考其 hot/cold pixel 检测和修复逻辑。
   链接: https://siril.readthedocs.io/en/latest/processing/cc.html
```

仿真方式：

```text
sparse_hot: 稀疏热像元
sparse_dead: 稀疏暗坏点
clustered_hot: 成簇热像元
dead_column: 坏列/坏行
mixed: 热点、暗点、坏列混合
```

level 控制：

```text
坏点数量
强度偏离
簇大小
是否出现坏列
```

输出 mask：

```text
M_dead_pixel: 坏点/热像元区域
```

### 4.4 Noise / smear

公开参考：

```text
1. Dynamic star map degradation model
   文献中使用 Gaussian noise 表示环境噪声、暗电流噪声和输出噪声，并讨论星图动态退化模型。
   链接: https://www.mdpi.com/2304-6732/9/10/673

2. CCD/CMOS 常规噪声建模
   可结合 read noise、dark current、shot noise、fixed-pattern noise。
```

采用策略：

```text
第一版继续使用当前代码中的 noise / smear 生成方法；
同时把参数含义对齐到 CCD/CMOS 噪声模型和动态星图退化文献。
```

原因是当前版本已经覆盖了实际需要的几类主要形式：随机读出噪声、光子噪声、暗电流噪声、背景起伏、列/行方向 smear。公开文献更多提供的是建模依据和参数范围，并不一定比当前代码更适合直接生成训练数据。更稳妥的做法是保留当前实现作为工程版本，然后用真实 clean / degraded 数据统计去校准各 level 的强度。

noise 仿真方式：

```text
gaussian noise: 读出噪声/环境噪声
poisson noise: 光子计数噪声
dark current noise: 暗电流相关噪声
background noise: 背景起伏
mixed noise: 多种噪声组合
```

smear 仿真方式：

```text
vertical column smear: 列方向拖影或偏置
horizontal row smear: 行方向拖影或偏置
bright-source smear: 强亮点导致局部列方向污染
mixed smear: 多种方向和强度组合
```

level 设计原则：

```text
level 1 基本接近 clean；
level 2 轻微可见；
level 3 明显；
level 4/5 对下游任务产生明显影响。
```

输出 mask：

```text
noise 一般可视为全图退化，不一定有局部 M_deg；
smear 需要 M_smear，记录被行/列结构影响区域。
```

### 4.5 Motion blur

公开参考：

```text
1. Image Degradation Model for Dynamic Star Maps in Multiple Scenarios
   使用 PSF、曝光分段和星点动态轨迹叠加模拟 motion blur。
   链接: https://www.mdpi.com/2304-6732/9/10/673

2. Motion modeling and blurred image simulation of the star tracker used for deep-space missions
   提出由 trajectory、intensity 和 PSF 三个描述符组成的 motion kernel。
   链接: https://opg.optica.org/josab/abstract.cfm?uri=josab-39-11-2934

3. Star Image Prediction and Restoration under Dynamic Conditions
   讨论角运动引起的星图模糊和恢复。
   链接: https://www.mdpi.com/1424-8220/19/8/1890
```

采用策略：

```text
当前版本的 linear / curved / non-uniform / jitter blur 可以作为第一版可控仿真；
论文版本建议进一步改成“曝光积分 + 星点轨迹 PSF”的实现。
```

更严格的 motion blur 生成应该模拟曝光期间星点在探测器上的连续运动。可以把一帧曝光拆成多个短曝光子帧，在每个子帧中根据角速度、抖动和 PSF 移动星点位置，最后积分得到模糊星图。这样比直接对整图做普通卷积更符合星点成像机理，也能保留背景和目标的不同运动表现。

仿真方式：

```text
linear blur: 匀速直线运动核
curved blur: 角速度变化导致曲线轨迹
non-uniform blur: 曝光期间速度变化
jitter blur: 姿态扰动导致局部抖动
```

输出记录：

```text
blur_kernel
blur_direction
trail_length
motion mask
```

---

## 5. 数据生成部分

数据设计目标：

```text
1. 提供可靠 paired restoration 监督。
2. 提供 star / target / background / degradation mask。
3. 支持 Light-DepictQA 的退化、风险、pairwise 训练。
4. 支持 Policy Net 的离线 reward 计算。
5. 支持 executor 从 single 到 multiple degradation 的逐步训练。
```

### 5.1 完全仿真数据

生成方式：

```text
星表或随机星场 -> clean star image
注入空间目标 -> clean image with target
添加 degradation -> degraded image
```

标签：

```text
clean GT
star mask
star centroid
target mask
target centroid / bbox / trajectory
background mask
degradation mask
degradation type / level / mode / seed
```

用途：

```text
1. 训练和验证 task-aware loss。
2. 计算 Star Preservation、Target Recall、Centroid Error。
3. 训练 Light-DepictQA 的 observation risk / damage risk。
4. 训练 Policy Net 的 GT-based reward。
```

优点：

```text
标签最完整，适合做指标和策略训练。
```

缺点：

```text
真实感可能不足，需要真实 clean 和真实退化数据校准。
```

### 5.2 半仿真数据

生成方式：

```text
真实或公开 clean_data + 人为添加 degradation
```

clean_data 来源可包括：

```text
TESS / Kepler target pixel files
Hubble / MAST 公开图像
自有相对干净空间图像
```

参考入口：

```text
TESSCut MAST: https://mast.stsci.edu/tesscut/docs/index.html
Kepler MAST: https://archive.stsci.edu/kepler/
Hubble Legacy Archive: 可通过 MAST/HLA 获取公开 HST 数据
```

标签：

```text
clean GT: 有
star mask: 可由星点检测器、星表匹配或伪标签生成
background mask: 可由 star/target mask 反推
目标 mask: 若原图没有目标，可注入仿真目标；若真实目标存在，需要人工或轨迹挖掘
退化 mask: 添加退化时记录
```

置信度体现：

```text
完全仿真 mask: confidence = 1.0
半仿真 star pseudo-mask: confidence = 0.6 - 0.9
人工校验 mask: confidence 可提高到 0.9 - 1.0
```

训练时可使用 confidence weighting：

```text
L_masked = confidence * L_region
```

用途：

```text
训练主要 executor，降低 simulation-to-real gap。
训练 Light-DepictQA 的退化等级和部分 risk 估计。
验证模型对真实背景的适应能力。
```

### 5.3 真实退化数据

来源：

```text
TESS / Kepler / Hubble / 自有真实退化图像
包含真实 stray light、真实背景异常、真实传感器噪声、真实坏点等
```

特点：

```text
通常没有 clean GT；
通常没有完整 star/target/degradation mask；
最能反映真实部署分布。
```

用途：

```text
1. 仿真退化模式校准。
2. 无参考真实域测试。
3. 人工主观评估。
4. Light-DepictQA 的弱监督/伪标签微调。
5. Policy Net 在线 proxy 稳定性验证。
```

不建议直接作为强监督 restoration 训练主数据，除非有可靠 paired GT 或可接受的 pseudo target。

### 5.4 Single / Double / Triple Degradation

数据必须支持三层退化：

```text
single degradation
double degradation
triple degradation
```

核心原则：

```text
每个多退化样本必须可追溯到上一层父图。
```

例如：

```text
clean A
AS = A + smear
AD = A + dqg
ASD = AS + 同一个 dqg 参数
```

这样可以构建 executor 训练目标：

```text
训练 denoising:
  input = noise + dqg
  target = dqg

训练 dehazing:
  input = noise + dqg
  target = noise

训练 decosmic:
  input = cosmic + noise + dqg
  target = noise + dqg
```

一句话：

```text
Single degradation teaches what to remove.
Multiple degradation teaches what not to remove.
```

---

## 6. Executor

### 6.1 Executor 的定位

Executor 是 STAR-Agent 中真正执行图像恢复的工具。最终在线部署不能依赖 SwinIR、Restormer、DehazeFormer 这类重模型，而应使用轻量 executor。

重模型的定位：

```text
1. baseline
2. upper bound
3. optional teacher
4. 离线工具能力评估参考
```

最终在线 executor 应尽量满足：

```text
小参数量
小 FLOPs
低内存
固定算子
INT8 量化友好
适合 tile/streaming 推理
可解释和可控
```

### 6.2 每类 executor 是否需要训练

子任务划分建议按照退化机理，而不是按照视觉上是否像“噪声”来划分：

```text
noise: 随机或统计型强度扰动，通常影响全图或大面积背景；
smear: 行/列方向结构性污染，通常和读出、电荷转移或强亮源串扰有关；
cosmic ray: 稀疏高能粒子击中探测器形成的局部异常；
dead pixels: 探测器固定缺陷，位置相对稳定；
stray light / dqg: 视场外光源或散射造成的低频背景场；
motion blur: 曝光期间相机/目标相对运动导致的 PSF 扩展。
```

因此 cosmic ray 和 dead pixel 不建议简单划归 denoising。它们虽然也表现为异常亮点或暗点，但修复方式更依赖 mask 定位、坏点表、局部插值和 inpainting，而不是全图随机噪声抑制。smear 也不建议完全并入普通 noise，因为它有明确的行/列结构先验，单独建模更容易轻量化，也更可解释。

| 子任务 | 退化 | 推荐 executor | 是否需要训练 | 说明 |
|---|---|---|---|---|
| `decosmic ray` | cosmic ray | L.A.Cosmic-like / median / astroscrappy baseline | 通常不需要 | 传统算法强且轻量 |
| `dead pixel repair` | dead pixels | bad pixel mask + interpolation / inpaint | 通常不需要 | 可直接星载部署 |
| `denoising` | random noise / background noise | TinyDenoise-Star / SwinIR-slim | 需要 | 需保护弱星点和目标 |
| `desmear` | smear | column/row profile + tiny refinement | 建议训练轻量 refinement | smear 有明显行/列结构，和随机噪声机理不同 |
| `stray light removal` | dqg / solar stray light | TinyStrayLightNet / background-field estimator | 需要 | 输出去杂散光后的图像，内部可先估计背景场 |
| `motion deblurring` | motion blur | Restormer-tiny / TinyDeblur-Star | 需要 | 针对星点拖尾、姿态抖动、曝光积分模糊 |

### 6.3 轻量化路线

三种路线并存：

```text
1. Prior-guided lightweight design
   根据退化机理直接设计小模型或传统算法。

2. Structured slimming of heavy architectures
   对 SwinIR / Restormer / DehazeFormer 做通道、深度、block、head 级压缩。

3. Optional task-aware distillation
   当 light model 直接训练不够好，且 heavy teacher 在下游指标上确实更好时，再蒸馏。
```

剪枝优先采用结构化剪枝：

```text
channel pruning
block pruning
depth slimming
width slimming
head pruning
```

不优先采用非结构化稀疏剪枝，因为 FPGA/边缘硬件不一定能有效加速。

### 6.4 各 executor backbone 建议

#### 6.4.1 Denoising

候选：

```text
TinyDenoise-Star
SwinIR-slim
MobileNet-UNet
Depthwise residual CNN
```

训练建议分成“全仿真预训练 + 半仿真适配 + 任务感知微调”三步：

```text
Stage 1: 全仿真 single-noise pretraining
Stage 2: 半仿真 single/double/triple degradation fine-tuning
Stage 3: task-aware loss fine-tuning
```

每一步的理由：

```text
Stage 1 的目的是让模型先学会最基础、最干净的噪声到 clean 映射。
全仿真数据有完整 GT、星点 mask、目标 mask、背景 mask，适合稳定训练低层恢复能力。

Stage 2 的目的是解决真实背景域偏移和多退化共存问题。
半仿真数据使用真实 clean_data 作为背景，再叠加可控 noise、dqg、smear、motion blur 等。
对 denoising executor 来说，多退化样本的 target 不是 clean，而是“只去掉 noise，保留其他退化”的父图。
这样模型不会错误地把 dqg、smear、motion blur 都当作 noise 一起去掉。

Stage 3 的目的是让恢复结果服务于下游任务。
在这一阶段加入星点保持、目标保持、光通量保持、质心偏移等 task-aware loss，避免模型为了提高 PSNR 过度平滑弱星点和弱目标。
```

推荐 loss：

```text
L_denoise = L_restore
          + λ_region * L_region
          + λ_flux   * L_flux
          + λ_bg     * L_bg
```

其中 `L_restore` 使用 Charbonnier 或 L1，`L_region` 提高星点和目标区域权重，`L_flux` 保护星点/目标亮度，`L_bg` 抑制背景残余噪声。

#### 6.4.2 Stray light / dqg

候选：

```text
TinyStrayLightNet
DehazeFormer-slim
background-field estimator
```

推荐输出形式：

```text
input -> predicted stray/background field
restored = input - field
```

原因：

```text
dqg / stray light 多为低频背景场。
背景场 field 指的是由杂散光、dqg 或背景梯度引入的额外低频亮度分量 B_stray(x, y)。
模型内部先估计 B_stray，再用 restored = input - B_stray 得到最终图像。
这样比直接从 input 回归 clean 更轻、更可解释，也更不容易改变星点和目标结构。
```

工程接口上，`stray light removal executor` 仍然应该输出去除杂散光后的空间图像，而不是只输出 field。field 可以作为额外 debug / explainability 输出保存：

```text
executor input: degraded image
executor output: restored image
optional output: estimated stray/background field
```

#### 6.4.3 Smear

候选：

```text
column/row profile estimator
TinySmearRefineNet
```

思路：

```text
先用列/行统计估计 smear bias；
再用小网络做 residual refinement。
```

#### 6.4.4 Motion blur

候选：

```text
Restormer-tiny
TinyDeblur-Star
PSF estimation + light deconvolution
```

如果第一篇工作量受限，可以先保留为扩展实验。

### 6.5 Executor 训练数据

不要只训练 single degradation。推荐：

```text
Stage 1: single degradation 100%
Stage 2: single : double : triple = 6 : 3 : 1
Stage 3: 增加 task-aware loss 和真实背景数据
```

对于某个 executor，多退化 target 必须是：

```text
移除当前 executor 负责的退化，保留其他退化。
```

### 6.6 Executor loss 设计

推荐总 loss：

```text
L_total = L_restore + λ_region * L_region + λ_bg * L_bg + λ_flux * L_flux + λ_heatmap * L_heatmap
```

基础项：

```text
L_restore = Charbonnier(restored, target)
```

区域项：

```text
L_region = w_target * ||M_target * (restored - target)||_1
         + w_star   * ||M_star   * (restored - target)||_1
         + w_deg    * ||M_deg    * (restored - target)||_1
         + w_bg     * ||M_bg     * (restored - target)||_1
```

其中：

```text
M_deg 是当前退化影响区域 mask。
它让模型重点修复真正被退化污染的区域。
```

Flux 项：

```text
Flux(restored, M) = sum(M * restored)
Flux(target, M) = sum(M * target)
L_flux = |Flux(restored, M_star/target) - Flux(target, M_star/target)| / (Flux(target, M) + eps)
```

含义：

```text
保护星点和目标的总亮度，避免去噪/去杂散光把弱星点或目标压暗。
```

背景项：

```text
L_bg_smooth = TV(M_bg * restored)
```

注意：

```text
L_bg_smooth 权重不能过大，否则可能过度平滑。
```

推荐第一版：

```text
L_total = 1.0 * L_charbonnier + 1.0 * L_region + 0.03 * L_bg_smooth + 0.1 * L_flux
```

如果使用 heavy model distillation：

```text
L_distill = ||M_safe * (student - teacher)||_1
```

其中 `M_safe` 通常选择背景区域，避免 student 在星点/目标区域模仿 teacher 的过平滑副作用。

---

## 7. Star-DepictQA-Lite

### 7.1 定位

Star-DepictQA-Lite 是 STAR-Agent 的状态感知模块。它不是在线 LLM，也不是开放式问答模型，而是一个空间图像专属、轻量化、多任务质量判别器。

它负责输出：

```text
degradation probability
severity level
observation risk
restoration damage risk
background residual score
pairwise quality ranking
```

这些输出构成 Policy Net 的重要 state。

### 7.2 Backbone 选择

推荐 backbone：

```text
MobileNetV3-small
ShuffleNetV2
EfficientNet-lite
Small CNN
TinyViT-lite，可选
```

如果目标是 FPGA：

```text
优先 Small CNN / MobileNet-like depthwise conv。
避免大 attention、复杂 LayerNorm 和动态 shape。
```

### 7.3 输入形式

第一版推荐：

```text
image -> lightweight encoder -> shared feature
prior vector -> MLP projection
concat(image feature, prior feature) -> multi-head outputs
```

prior vector 可包含：

```text
star_count
background_std
background_gradient
artifact_area_ratio
target_candidate_count
average_star_snr
bright_pixel_ratio
```

也可以做多通道输入：

```text
image + star_candidate_mask + background_map + artifact_candidate_mask + target_heatmap
```

但第一版推荐图像特征 + 结构化特征，因为更轻、更可解释。

### 7.4 输出 head

建议包含：

```text
1. Degradation classification head
   多标签分类，输出每类退化 prob。

2. Severity estimation head
   每类退化 1-5 level 或 very low 到 very high。

3. Observation risk head
   用于原始 degraded image。
   输出 star_observation_risk、target_observation_risk、background_interference_risk。

4. Restoration damage head
   用于 restored image。
   输出 star_damage_risk、target_damage_risk、background_residual_score。

5. Pairwise ranking head
   输入两个候选恢复图，判断哪个更适合下游任务。
```

### 7.5 Risk 标签如何计算

#### 7.5.1 Observation risk

用于原始 degraded image。

离线用：

```text
degraded image vs clean/GT
star mask / target mask / background mask
```

计算：

```text
star_observation_risk = a * star_missing_score + b * star_flux_error + c * centroid_error + d * star_snr_drop

target_observation_risk = a * target_missing_score + b * target_flux_error + c * centroid_error + d * target_snr_drop

background_interference_risk = a * bg_error + b * bg_gradient_error + c * bg_noise_increase
```

意义：

```text
原始退化图对下游星点/目标观测造成多大风险。
```

#### 7.5.2 Restoration damage risk

用于 executor 输出的 restored image。

离线需要先调用 executor 生成 restored candidates：

```text
degraded -> executor -> restored
```

然后用：

```text
restored image vs clean/target
star mask / target mask
```

计算：

```text
star_damage_score = a * (1 - StarPreservationRate) + b * StarFluxError + c * StarCentroidError + d * StarShapeError

target_damage_score = a * (1 - TargetRecall) + b * TargetFluxError + c * TargetCentroidError + d * ConfidenceDrop

background_residual_score = a * background_error + b * background_gradient_error
```

训练时输入仍然只有图像和可选 prior，不输入 GT。

GT 只用于离线生成 label。

### 7.6 训练数据构建

数据分三类：

```text
1. degraded image -> degradation type / severity / observation risk
2. restored image -> residual degradation / damage risk
3. restored image A vs B -> pairwise quality ranking
```

meta 示例：

```json
{
  "image_A": "restored/swinir/img_001.png",
  "query": "What is the star damage risk in this restored space image?",
  "answer": "low"
}
```

pairwise 示例：

```json
{
  "image_A": "restored/tiny/img_001.png",
  "image_B": "restored/swinir/img_001.png",
  "query": "Which image better preserves stars and weak targets?",
  "answer": "Image B"
}
```

### 7.7 Loss

```text
L = λ_cls * L_cls + λ_level * L_level + λ_risk * L_risk + λ_rank * L_rank
```

其中：

```text
L_cls: BCE multi-label classification
L_level: CrossEntropy 或 ordinal loss
L_risk: MSE / SmoothL1
L_rank: CrossEntropy 或 margin ranking
```

### 7.8 生成式 DepictQA 多任务干扰现象

在当前星空版 DepictQA 微调实验中观察到一个重要现象：`single severity assessment` 和 `pairwise quality comparison` 虽然都属于图像质量评估任务，但二者的输出空间差异很大，混合训练时可能产生明显任务干扰。

具体表现是：

```text
single severity task:
  task_type = quality_single_A_noref
  期望输出 = very low / low / medium / high / very high

pairwise comparison task:
  task_type = quality_compare_noref
  期望输出 = Image A / Image B
```

实验中发现，即使 `single` 样本数量约为 200 多万、`compare` 样本数量约为 170 多万，生成式 DepictQA 仍可能在 `quality_single_A_noref` 下输出 `Image A`。这说明问题不只是样本数量比例，而是 compare 任务的短答案空间和强监督信号会给模型带来更强的输出先验。

这个现象的含义：

```text
1. task_type 可以规定输入格式和 system prompt，但不能硬约束生成式模型的输出空间。
2. compare 任务只有 Image A / Image B 两类答案，学习难度低、输出模式强。
3. single 任务有 5 档等级，且大量样本可能是 very low，真实有效监督更稀疏。
4. 多任务混合训练如果没有任务平衡或输出约束，容易出现输出空间混淆。
```

因此，这个现象不宜单独作为主创新点，但可以作为 STAR-Agent 中质量判别模块设计的重要动机：

```text
生成式 VQA 模型在星空 restoration agent 中承担状态感知角色时，
需要额外考虑任务间输出空间干扰；
不能简单把 single severity 和 pairwise comparison 样本拼接后直接微调。
```

推荐解决策略：

```text
1. 阶段式训练
   Stage 1: 只训练 single severity，先保证退化识别和等级评估稳定；
   Stage 2: 加入少量 compare 样本，保持 single 为主；
   Stage 3: 同时监控 single invalid answer rate 和 compare accuracy。

2. 任务平衡采样
   不按原始样本数量直接拼接；
   每个 batch 或每个 epoch 控制 single : compare 比例，例如 3:1 或 5:1。

3. 输出空间约束
   single 任务只允许输出 5 个 severity label；
   compare 任务只允许输出 Image A / Image B；
   可以通过 candidate scoring、logit masking 或后处理重试实现。

4. 结构化轻量模型
   在 Star-DepictQA-Lite 中不再依赖开放式文本生成；
   而是设计 task-specific heads：
   degradation head、severity head、risk head、comparison head。
```

建议在后续实验中单独报告：

```text
Single Accuracy
Compare Accuracy
Single Invalid Answer Rate
Single-to-Compare Confusion Rate
```

其中：

```text
Single Invalid Answer Rate =
  single 任务中输出不属于 5 个 severity label 的比例。

Single-to-Compare Confusion Rate =
  single 任务中错误输出 Image A / Image B 的比例。
```

这个指标可以直接支撑“为什么需要 Star-DepictQA-Lite 的结构化多头设计”，也能解释为什么最终星上部署不应依赖开放式生成模型承担关键状态判别。

---

## 8. 下游任务反馈

第一篇建议接两个下游任务。

### 8.1 星点检测 / 质心定位 / 星图匹配

推荐工具：

```text
LOST: Open-source Star Tracker
链接: https://github.com/UWCubeSat/lost

tetra3rs / tetra3:
支持 centroid extraction 和 lost-in-space plate solving。
链接: https://tetra3rs.dev/tutorials/basic-solve/

Astrometry.net:
开源 plate solving 系统，可作为离线强 baseline。
链接: https://astrometry.net/doc/
```

在线 proxy 指标，不需要 GT：

```text
plate_solve_success
solver_confidence
matched_star_count
detected_star_count
unmatched_candidate_ratio
reprojection_error
centroid_fit_residual
star_snr_mean
star_fwhm_mean
```

离线评价指标，需要 GT 或星表匹配参考：

```text
Star Detection Recall
False Star Rate
Centroid Error
Plate-solve Success
Matched Star Count
Reprojection Error
```

### 8.2 小目标检测

如果目标是条纹目标，可参考：

```text
ASTRiDE: Automated streak detection for astronomical images
链接: https://github.com/dwkim78/ASTRiDE
```

如果目标是点状弱目标，建议先自建轻量 detector：

```text
local background subtraction
LoG / blob detector
connected component
SNR threshold
sequence consistency，可选
```

当前第一阶段 clean 仿真中，目标按两类检测器需求共同设计：

```text
image-level target policy:
  20% 图像无目标
  80% 图像含目标
  含目标图像随机注入 1-5 个目标
  无目标图像用于估计目标检测器 false alarm

point_blob target: 30%
  下游检测器: 轻量 LoG / blob detector
  目标形态: 接近星点的弱小 blob，可有轻微椭圆
  设计目的: 覆盖点状空间小目标，但单帧语义区分较难

short_streak target: 70%
  下游检测器: ASTRiDE 或类 ASTRiDE streak detector
  目标形态: 短条纹/拖尾，具有方向、长度和长宽比
  设计目的: 条纹目标几何先验明确，更适合第一阶段稳定下游反馈
```

这意味着下游目标检测不是一个模型强行处理所有目标，而是两个 detector family 共用统一标签接口：

```text
target_mask
target_center
bbox
target_type
detector_family
snr
```

如果 `target_type = point_blob`，评价时使用 LoG/blob detector 的输出。
如果 `target_type = short_streak`，评价时使用 ASTRiDE/streak detector 的输出。
这样可以避免“没有退化也检测不出目标”的问题，使下游反馈真正反映 restoration 是否保护了目标。

在线 proxy：

```text
target_candidate_count
target_candidate_snr
target_detector_confidence
temporal_consistency，如果有序列
tracklet_length，如果有序列
motion_model_residual，如果有序列
```

离线评价：

```text
Target Recall
Precision
False Alarm Rate
mAP
Weak Target Recall
Target Centroid Error
```

### 8.3 在线和离线的区别

在线无 GT：

```text
只能使用下游工具输出的 proxy feedback。
不能得到真实 recall / false alarm / centroid error。
```

离线有 GT：

```text
可以用 GT 计算真实 reward 和 evaluation metrics。
但 GT 不能作为 Policy Net 输入。
```

---

## 9. Policy Net

Policy Net 是 STAR-Agent 的核心规划器。

### 9.1 定位与设计动机

它承担：

```text
选择下一步 subtask
选择 executor tool
判断是否 stop
判断是否 rollback
估计继续恢复的收益
```

它不是在线 LLM，而是轻量结构化决策网络。

这样设计的核心动机是：

```text
1. LLM 擅长离线探索、解释和归纳经验，但在线部署太重。
2. 空间平台需要稳定、低功耗、可验证的决策器。
3. restoration 决策本质上可以被表示为结构化状态到动作的映射。
4. 离线阶段可以用 LLM + GT + 强 executor 生成高质量决策样本。
5. 在线阶段只保留 Policy Net、轻量 executor、Star-DepictQA-Lite 和少量经验统计。
```

因此，Policy Net 不是从零随机试错，而是学习离线 LLM / oracle 在大量图像、退化组合和工具组合上总结出来的策略。

可以把它理解为：

```text
LLM / oracle:
  地面老师，负责探索、比较、解释和生成训练标签。

Policy Net:
  星上学生，负责用固定输入、固定输出、低算力方式复现老师的决策能力。
```

候选模型：

```text
第一版: MLP-based multi-head policy network
第二版: GRU-based sequence policy
第三版: Tiny Transformer-based policy
```

第一篇论文建议从 `MLP-based multi-head policy network` 开始。

原因是：

```text
1. 输入主要是结构化特征，不是长文本。
2. 动作空间有限，包括 stop、rollback 和若干 restoration tool。
3. MLP 参数量小、部署简单、便于 FPGA / 边缘平台移植。
4. 多 head 可以把 action、tool、stop、rollback、Q-value 分开监督，解释性更好。
```

如果后续要显式建模长历史，例如连续 5 步以上的工具序列，可以升级为 GRU 或 Tiny Transformer。

### 9.2 Policy Net 本体结构

第一版推荐使用共享 backbone + 多个输出 head：

```text
state vector s_t
    ↓
MLP Encoder / Tiny Backbone
    ↓
shared hidden feature z_t
    ├── action head
    ├── tool head
    ├── stop head
    ├── rollback head
    ├── Q head
    └── risk / gain auxiliary head，可选
```

其中：

```text
s_t:
  当前图像、历史步骤、下游 proxy、经验统计共同构成的状态向量。

z_t:
  Policy Net 对当前 restoration 状态的内部表征。

action head:
  预测下一步做什么 subtask，例如 denoise、dehaze、decosmic、deblur、stop。

tool head:
  在当前 subtask 下选择具体 executor，例如 SwinIR、MPRNet、DehazeFormer、astroscrappy。

stop head:
  判断是否已经可以停止 restoration。

rollback head:
  判断上一步工具是否伤害了星点/目标，是否需要回退。

Q head:
  估计每个候选动作的期望收益，用于更稳健地选择动作。

auxiliary head:
  可选，用于预测风险变化或下游收益，帮助 backbone 学到任务相关表征。
```

动作 head 和 Q head 的区别：

```text
action head:
  学 LLM / oracle 最终选择了什么动作。

Q head:
  学每个候选动作分别有多好。
```

所以，action head 更像“模仿老师的最终选择”，Q head 更像“学会给每个工具打分”。

### 9.3 Policy Net 输入

输入是固定长度 state vector，可分为八类。

总体写成：

```text
s_t = concat(
    f_deg_t,
    f_risk_t,
    f_downstream_t,
    f_image_t,
    f_history_t,
    f_delta_t,
    f_memory_t,
    f_retrieval_t
)
```

其中：

```text
f_deg_t:
  Star-DepictQA-Lite 输出的退化类型、存在概率、严重程度。

f_risk_t:
  星点、目标、背景被当前退化影响或被继续修复损伤的风险。

f_downstream_t:
  星图匹配、小目标检测、质心定位等下游模块的在线 proxy。

f_image_t:
  不依赖 GT 的图像统计量。

f_history_t:
  已经执行过哪些工具、当前第几步、还剩多少预算。

f_delta_t:
  上一步动作带来的变化量。

f_memory_t:
  经验库聚合出来的工具统计。

f_retrieval_t:
  可选的结构化 RAG 检索特征。
```

为了避免实现时混淆，需要明确每类 feature 的来源。

| feature 组 | 主要来源 | 在线是否可用 | 是否直接依赖 GT | 说明 |
|---|---|---|---|---|
| `f_deg_t` | Star-DepictQA-Lite / 退化判别器 | 是 | 否 | 判断当前图像有哪些退化，以及严重程度 |
| `f_risk_t` | Star-DepictQA-Lite risk head / 图像统计 / 下游 proxy / 轻量风险估计器 | 是 | 否 | 表示当前图像或继续恢复对星点、目标、背景的风险 |
| `f_downstream_t` | LOST、tetra3rs、星点检测器、小目标检测器 | 是 | 否 | 在线下游任务反馈，只是 proxy，不是真实 GT 指标 |
| `f_image_t` | image statistics extractor | 是 | 否 | 从当前图像直接计算的背景、亮点、梯度、对比度等统计 |
| `f_history_t` | agent runtime state / execution trace | 是 | 否 | 当前 episode 内部记录的历史动作和预算 |
| `f_delta_t` | 当前 step 与上一 step 的 feature 差值 | 是 | 否 | 由 `state_t - state_{t-1}` 得到，反映上一步工具效果 |
| `f_memory_t` | offline_verified_memory / online_unverified_memory 聚合统计 | 是 | 否 | 历史经验压缩后的工具成功率、收益、损伤率、耗时 |
| `f_retrieval_t` | 结构化经验库检索模块 | 可选 | 否 | 从相似历史 case 中提取的固定维度检索特征 |

注意：

```text
GT、clean image、真实 mask、真实下游 label 只能用于离线计算 reward 和生成 teacher label。
它们不能作为 Policy Net 的输入 feature。
```

因此，Policy Net 的输入必须满足：

```text
训练时可获得；
在线部署时也可获得；
不直接包含 clean GT 或真实标签；
但可以通过离线 reward 让模型间接学到 GT 监督过的策略。
```

#### 9.3.1 Degradation features

```text
noise_prob, noise_level
dqg_prob, dqg_level
solar_stray_light_prob, solar_stray_light_level
smear_prob, smear_level
cosmic_ray_prob, cosmic_ray_level
dead_pixels_prob, dead_pixels_level
motion_blur_prob, motion_blur_level
```

这些特征来自 Star-DepictQA-Lite。

具体来源：

```text
模块：
  Star-DepictQA-Lite degradation head

输入：
  当前图像 I_t

输出：
  每一类 degradation 的存在概率 prob 和严重程度 level

在线：
  直接运行轻量 Star-DepictQA-Lite 得到。

离线训练：
  可以用原始 DepictQA / 强判别器 / 仿真标签辅助监督 Star-DepictQA-Lite；
  但训练 Policy Net 时，输入仍然使用判别器输出，而不是直接喂 GT 标签。
```

其中：

```text
degradation_prob:
  表示某类退化存在的概率或置信度。

degradation_level:
  表示该退化的严重程度，可以用 0-5 或归一化到 [0, 1]。
```

一个重要细节是：

```text
level 不是简单的视觉等级，而是和后续工具调度相关的状态变量。
```

例如：

```text
noise_level 高:
  更倾向先执行 denoising。

dqg_level 高但 noise_level 也高:
  需要结合经验判断先 denoise 还是先 dehaze。

cosmic_ray_level 高:
  优先执行 decosmic_ray，因为局部尖峰会干扰星点检测和后续低频估计。
```

#### 9.3.2 Risk features

```text
star_observation_risk
target_observation_risk
background_interference_risk
star_damage_risk
target_damage_risk
background_residual_score
```

绝对 risk 表示当前状态危险程度。

这里的 risk 不是 GT，而是在线可估计的风险。

具体来源：

```text
模块 1：
  Star-DepictQA-Lite risk head

模块 2：
  image statistics extractor

模块 3：
  downstream proxy module

模块 4，可选：
  lightweight damage-risk predictor
```

各类 risk 可以这样获得：

```text
star_observation_risk:
  来自星点检测数量、匹配星数量、reprojection error、背景噪声和运动模糊等级。

target_observation_risk:
  来自目标候选 SNR、目标检测置信度、背景残差、杂散光覆盖区域和运动模糊等级。

background_interference_risk:
  来自背景均值、背景标准差、背景梯度、dqg / solar stray light level。

star_damage_risk:
  来自 Star-DepictQA-Lite risk head 或轻量风险预测器；
  表示如果继续执行 restoration，弱星点可能被抹掉、质心偏移或亮度改变。

target_damage_risk:
  来自 target risk head 或轻量风险预测器；
  表示小目标可能被去噪、去杂散光或去模糊工具误伤。

background_residual_score:
  来自背景统计、低频残差估计、degradation head 对 dqg / solar stray light / smear 的判断。
```

离线阶段可以用真实 mask 和 GT 指标训练这些 risk head：

```text
例如用 Star Preservation Rate、Target Recall、Flux Error、Centroid Shift 生成 risk label。
```

但在线阶段：

```text
只能使用 risk head 的预测值和 proxy 统计；
不能使用真实 Star Preservation 或真实 Target Recall。
```

例如：

```text
star_observation_risk:
  当前图像中星点被噪声、杂散光、运动模糊干扰的风险。

target_observation_risk:
  当前图像中小目标被退化覆盖、抹除或伪目标混淆的风险。

star_damage_risk:
  如果继续执行某些工具，可能损伤弱星点的风险。

target_damage_risk:
  如果继续执行某些工具，可能损伤小目标的风险。

background_residual_score:
  当前背景残余退化程度，尤其用于 dqg / solar stray light / smear。
```

risk 的绝对值可以回答：

```text
现在是否危险？
```

risk 的变化量可以回答：

```text
上一步工具是改善了风险，还是造成了副作用？
```

因此 Policy Net 输入中既要有 `current risk`，也要有 `risk delta`。

#### 9.3.3 Downstream proxy features

```text
plate_solve_success
solver_confidence
matched_star_count_norm
detected_star_count_norm
unmatched_candidate_ratio
reprojection_error_norm
centroid_fit_residual_norm
target_candidate_snr
target_confidence
```

这些指标在线不需要 GT，可以由 LOST、tetra3rs、星点检测器或小目标检测器输出。

具体来源：

```text
plate_solve_success:
  来自 LOST / tetra3rs / astrometry solver 是否成功求解。

solver_confidence:
  来自星图匹配器的匹配置信度、残差评分或内部 score。

matched_star_count_norm:
  来自星图匹配成功的星点数量，并用图像尺寸、预期星点数量或验证集统计归一化。

detected_star_count_norm:
  来自星点检测器检测到的候选星点数量。

unmatched_candidate_ratio:
  来自 detected stars 与 matched stars 的差值比例；
  值高时可能说明伪星点增加。

reprojection_error_norm:
  来自星图匹配后的重投影误差。

centroid_fit_residual_norm:
  来自星点质心拟合残差，例如 Gaussian fitting residual。

target_candidate_snr:
  来自小目标检测候选区域的局部信噪比。

target_confidence:
  来自小目标检测器输出的置信度。
```

这些特征的关键点是：

```text
它们是在线 proxy，不是最终真实下游指标。
```

例如在线可以知道：

```text
当前 plate solve 是否成功；
候选星点数量是否异常；
目标检测置信度是否上升。
```

但在线不知道：

```text
真实 Star Recall；
真实 Target Recall；
真实 False Alarm Rate。
```

它们的作用不是替代 GT 指标，而是在线判断：

```text
restoration 后星图匹配是否更容易？
星点候选是否更稳定？
目标候选信噪比是否更高？
是否产生了大量伪星点？
```

例如：

```text
matched_star_count_norm 上升:
  说明星点可匹配性增强。

unmatched_candidate_ratio 上升:
  可能产生了伪星点。

reprojection_error_norm 下降:
  说明星图匹配质量更好。

target_candidate_snr 上升:
  说明弱目标更容易被检测。
```

#### 9.3.4 Image statistics

```text
background_mean
background_std
background_gradient
background_nonuniformity
bright_pixel_ratio
saturated_pixel_ratio
artifact_area_ratio
local_contrast
```

这些是纯图像统计量，不依赖 GT。

具体来源：

```text
模块：
  image statistics extractor

输入：
  当前图像 I_t

实现方式：
  传统图像处理和统计计算，不需要神经网络。
```

各项来源：

```text
background_mean:
  通过背景区域估计得到。在线没有真实背景 mask 时，可以用低亮度分位数、sigma clipping 或星点剔除后的区域估计。

background_std:
  用背景候选区域的标准差估计噪声水平。

background_gradient:
  对背景低频图做水平/垂直梯度或多项式拟合残差估计。

background_nonuniformity:
  用低频背景图的最大最小差、分块均值方差或拟合残差衡量。

bright_pixel_ratio:
  统计超过高亮阈值的像素比例，可用于 cosmic ray、hot pixel、边缘强光源。

saturated_pixel_ratio:
  统计接近传感器饱和值的像素比例。

artifact_area_ratio:
  由异常亮点、异常列/行、杂散光区域估计得到的异常覆盖面积比例。

local_contrast:
  用局部窗口均值/方差、LoG 响应或星点候选局部对比度估计。
```

离线有 mask 时，可以用真实 `M_bg`、`M_deg` 做更准的统计；但训练 Policy Net 时建议模拟在线条件：

```text
优先使用在线可获得的估计 mask 或无 mask 统计；
避免训练时依赖真实 mask，导致部署时分布不一致。
```

它们可以补充 Star-DepictQA-Lite 没有显式输出的信息，例如：

```text
background_gradient:
  适合描述 dqg / solar stray light 的大尺度背景变化。

bright_pixel_ratio:
  适合描述 cosmic ray、hot pixel、饱和边缘光源。

artifact_area_ratio:
  适合描述退化覆盖面积。

local_contrast:
  可以辅助判断星点和背景是否被过度平滑。
```

#### 9.3.5 History features

```text
step_index
remaining_steps
last_action_id
last_tool_id
has_denoised
has_dehazed
has_decosmic
has_dead_pixel_repaired
has_deblurred
has_desmeared
num_tools_used
```

history features 的设计动机是防止重复执行工具或无效循环。

具体来源：

```text
模块：
  agent runtime controller / execution trace

输入：
  当前 restoration episode 的历史记录

输出：
  当前第几步、上一步动作、上一步工具、哪些 subtask 已经执行过、剩余预算。
```

各项来源：

```text
step_index:
  当前执行到第几步，从 0 开始计数。

remaining_steps:
  max_steps - step_index。

last_action_id:
  上一步执行的 subtask，例如 denoising / dehazing / stop / rollback。

last_tool_id:
  上一步具体工具，例如 SwinIR / MPRNet / DehazeFormer / astroscrappy。

has_denoised 等 flag:
  当前 episode 中是否已经执行过对应 subtask。

num_tools_used:
  当前已经调用过多少次 executor。
```

这些 feature 不来自图像内容，而来自 agent 自己的执行日志。

例如：

```text
如果 has_denoised = 1 且 delta_noise_level 很小:
  不应反复调用 denoising。

如果 last_tool_id 对 star_damage_risk 造成明显上升:
  rollback head 应该提高回退概率。

如果 step_index 接近最大步数:
  stop head 应该更谨慎评估继续收益。
```

#### 9.3.6 Delta features

每一步 restoration 后都要重新计算：

```text
delta_total_degradation_score
delta_noise_level
delta_dqg_level
delta_star_risk
delta_target_risk
delta_background_residual
delta_solver_confidence
delta_matched_star_count
delta_reprojection_error
last_estimated_task_gain
last_runtime_cost
```

delta features 的计算方式是：

```text
delta_x_t = x_t - x_{t-1}
```

具体来源：

```text
模块：
  state differencer / feedback evaluator

输入：
  state_{t-1}
  state_t
  上一步 action/tool/runtime

输出：
  当前状态相对上一状态的变化量。
```

例如：

```text
delta_noise_level:
  noise_level_t - noise_level_{t-1}

delta_dqg_level:
  dqg_level_t - dqg_level_{t-1}

delta_star_risk:
  star_risk_t - star_risk_{t-1}

delta_solver_confidence:
  solver_confidence_t - solver_confidence_{t-1}

last_runtime_cost:
  上一步 executor 的实际耗时，由 runtime controller 记录。
```

为了统一“越大越好”或“越小越好”的方向，建议对关键变化定义成 gain：

```text
degradation_gain_t = degradation_score_{t-1} - degradation_score_t
risk_gain_t        = risk_score_{t-1} - risk_score_t
proxy_gain_t       = downstream_proxy_t - downstream_proxy_{t-1}
```

例如：

```text
delta_noise_level < 0:
  噪声等级下降，是正向效果。

delta_star_risk > 0:
  星点风险上升，是负向副作用。

delta_solver_confidence > 0:
  星图匹配信心提升，是正向下游反馈。
```

#### 9.3.7 Tool memory features

对每个工具统计：

```text
tool_success_rate_current_combo
tool_avg_reward_current_combo
tool_star_damage_rate_current_combo
tool_target_gain_current_combo
tool_background_gain_current_combo
tool_runtime_mean
tool_failure_rate
```

这些来自历史 rollout 日志。

具体来源：

```text
模块：
  experience memory aggregator

输入：
  offline_verified_memory
  online_unverified_memory，可选，只能作为 proxy 统计

聚合键：
  degradation combo
  severity bin
  action / tool
  history pattern，可选

输出：
  当前退化组合下，各工具的成功率、平均 reward、损伤率、收益和耗时。
```

例如：

```text
tool_success_rate_current_combo:
  在类似 noise_l4 + dqg_l3 的 verified memory 中，该工具成功样本数 / 总样本数。

tool_avg_reward_current_combo:
  verified memory 中该工具的平均 GT-based reward。

tool_star_damage_rate_current_combo:
  verified memory 中该工具导致 Star Preservation 下降超过阈值的比例。

tool_target_gain_current_combo:
  verified memory 中该工具带来的 Target Recall 或 target proxy 平均提升。

tool_background_gain_current_combo:
  该工具降低背景残差的平均幅度。

tool_runtime_mean:
  历史执行耗时均值。

tool_failure_rate:
  工具执行失败、输出异常或触发 rollback 的比例。
```

需要注意：

```text
offline_verified_memory 的统计可信度高，可以用于训练和部署。
online_unverified_memory 只能作为在线 proxy 统计，最好降低权重或等待地面复核后再升级。
```

设计动机：

```text
同一个工具在不同退化组合上的效果不同。
```

例如：

```text
denoising 在 noise + dqg 中可能应该先执行；
但在 solar stray light 很强且弱目标靠近边缘时，强去噪可能损伤目标。
```

因此 Policy Net 不只看当前图像，还要看历史统计：

```text
在类似退化组合下，这个工具过去成功率如何？
平均收益如何？
是否容易伤害星点或目标？
耗时是否可接受？
```

#### 9.3.8 Retrieved experience features，可选 RAG

如果启用结构化 RAG：

```text
retrieved_best_action_id
retrieved_best_order_id
retrieved_avg_reward
retrieved_max_reward
retrieved_failure_rate
retrieved_similarity_mean
retrieved_topk_action_histogram
```

这些特征来自相似历史案例检索，而不是自然语言 prompt。

具体来源：

```text
模块：
  structured experience retriever

输入：
  当前 state 的 query vector
  offline_verified_memory 中的历史 case embedding
  online_unverified_memory，可选，低权重

输出：
  top-k 相似 case 的压缩统计特征。
```

query vector 可以由以下内容组成：

```text
degradation levels
risk features
downstream proxy features
image statistics
history features
```

检索后不直接把自然语言经验喂给 Policy Net，而是压缩成固定维度：

```text
retrieved_best_action_id:
  top-k case 中平均 reward 最高的下一步 action。

retrieved_best_order_id:
  top-k case 中平均 final reward 最高的完整恢复顺序。

retrieved_avg_reward:
  top-k case 的平均 reward。

retrieved_max_reward:
  top-k case 中最高 reward。

retrieved_failure_rate:
  top-k case 中失败、rollback 或负 reward 的比例。

retrieved_similarity_mean:
  top-k case 与当前 state 的平均相似度。

retrieved_topk_action_histogram:
  top-k case 中各类 action 出现频率。
```

因此，RAG 在这里不是：

```text
读取一段文字经验，然后让 LLM 推理。
```

而是：

```text
检索相似结构化案例，再把案例统计压缩成 Policy Net 可用的数值 feature。
```

### 9.4 不用 RAG 怎么做

不用 RAG 时，把历史经验压缩成固定字段：

```text
按 degradation combo + tool 聚合 success rate、avg reward、damage rate、runtime。
```

在线根据当前 combo 查表，将统计特征拼进 state。

优点：

```text
部署简单，固定维度，FPGA 友好。
```

缺点：

```text
经验被平均化，对罕见样本不够灵活。
```

### 9.5 用 RAG 怎么做

这里的 RAG 是结构化 case retrieval，不是语言 RAG。

经验库样本：

```json
{
  "state_key": {
    "noise_level": 5,
    "dqg_level": 3,
    "target_risk": 0.7
  },
  "action_sequence": ["denoising@SwinIR", "dehazing@DehazeFormer"],
  "final_reward": 0.20,
  "star_damage": 0.05,
  "failure": false
}
```

在线：

```text
当前 state -> query vector -> 检索 top-k 相似历史案例 -> 编码为固定 retrieved features -> 拼入 Policy Net 输入。
```

如果在线部署后积累新经验，可以周期性更新经验库，再在地面重新蒸馏或更新统计表。

### 9.6 Policy Net 输出与多 head 设计

```text
action_logits: 下一步 subtask/action
tool_logits: 当前 subtask 下选择哪个 tool
stop_prob: 是否停止
rollback_prob: 是否回退到上一步结果
q_values: 每个候选动作的预期收益
value: 当前状态整体继续恢复的预期收益，可选
risk_delta_pred: 执行动作后风险变化预测，可选
```

动作空间示例：

```text
0: stop
1: denoising@TinyDenoise
2: dehazing@TinyStrayLightNet
3: decosmic_ray@classic
4: dead_pixel_repair@classic
5: desmear@TinySmear
6: motion_deblurring@TinyDeblur
7: rollback
```

具体计算：

```text
z_t = Encoder(s_t)

π_action(a | s_t) = softmax(W_action z_t + b_action)

π_tool(k | s_t, a_t) = softmax(W_tool z_t + b_tool)

p_stop = sigmoid(W_stop z_t + b_stop)

p_rollback = sigmoid(W_rollback z_t + b_rollback)

Q(s_t, a) = W_q z_t + b_q
```

其中：

```text
π_action:
  给出下一步动作概率。

π_tool:
  给出工具选择概率。

p_stop:
  给出停止概率。

p_rollback:
  给出回退概率。

Q(s_t, a):
  给出每个动作的期望收益估计。
```

最终决策时可以组合 action head 和 Q head：

```text
score(a) = log π_action(a | s_t) + η * Q_norm(s_t, a) - μ * cost(a) - ν * risk_penalty(a)

a_t = argmax legal_action score(a)
```

这样做的动机是：

```text
action head 保留 LLM / oracle 的策略习惯；
Q head 提供每个动作的收益估计；
cost 和 risk penalty 保证在线部署可控、安全。
```

### 9.7 Reward 设计与计算 ，这些reward都是利用GT 计算出来的， 主要是动作， 回滚， 停止 这三种reward

Reward 是离线训练阶段评价动作好坏的分数，不是神经网络的 loss。

二者关系是：

```text
reward:
  用 GT、mask、下游任务结果计算，表示某个动作或序列到底好不好。

loss:
  用来训练 Policy Net，让它预测 LLM / oracle 的动作，或者预测 reward / Q value。
```

也就是说：

```text
reward 是监督信号的来源；
loss 是优化 Policy Net 参数的数学目标。
```

#### 9.7.1 单步 reward

对于状态 `s_t` 下执行动作 `a_t`，得到恢复图像：

```text
I_{t+1} = Tool_{a_t}(I_t)
```

离线阶段因为有 clean GT、mask、退化参数和下游标注，可以计算：

```text
OARS_t, OARS_{t+1}
TargetRecall_t, TargetRecall_{t+1}
StarPreservation_t, StarPreservation_{t+1}
BackgroundResidual_t, BackgroundResidual_{t+1}
DegradationResidual_t, DegradationResidual_{t+1}
Runtime(a_t)
```

定义各项增益：

```text
ΔOARS = OARS_{t+1} - OARS_t

ΔTarget = TargetRecall_{t+1} - TargetRecall_t

ΔStar = StarPreservation_{t+1} - StarPreservation_t

ΔDeg = DegradationResidual_t - DegradationResidual_{t+1}

ΔBg = BackgroundResidual_t - BackgroundResidual_{t+1}
```

注意：

```text
OARS、TargetRecall、StarPreservation 越大越好，所以用 after - before。
DegradationResidual、BackgroundResidual 越小越好，所以用 before - after。
```

再定义损伤项：

```text
DamageStar = max(0, StarPreservation_t - StarPreservation_{t+1})

DamageTarget = max(0, TargetRecall_t - TargetRecall_{t+1})

FalseStarIncrease = max(0, FalseStarRate_{t+1} - FalseStarRate_t)

FalseTargetIncrease = max(0, FalseTargetRate_{t+1} - FalseTargetRate_t)
```

单步 reward 可以写成：

```text
r_t =
    α_oars   * ΔOARS
  + α_target * ΔTarget
  + α_star   * ΔStar
  + α_deg    * ΔDeg
  + α_bg     * ΔBg
  - β_star   * DamageStar
  - β_target * DamageTarget
  - β_falseS * FalseStarIncrease
  - β_falseT * FalseTargetIncrease
  - β_time   * RuntimeNorm(a_t)
  - β_step   * StepPenalty
```

推荐权重关系：

```text
β_target 和 β_star 要偏大，防止 restoration 为了提升背景质量而损伤弱目标和星点。
α_target 和 α_star 应该高于 α_bg，避免大面积背景主导 reward。
β_time 根据部署平台算力预算设置。
```

一个直观例子：

```text
某次 denoising 后：
  背景噪声下降，ΔBg > 0；
  星点保持率基本不变，DamageStar ≈ 0；
  小目标召回上升，ΔTarget > 0；
  runtime 可接受。

则 reward 为正，说明该动作值得学习。

如果 denoising 后：
  PSNR 上升；
  但弱星点被抹掉，DamageStar 很大；
  小目标召回下降，DamageTarget 很大。

则 reward 可能为负，说明该动作不应被 Policy Net 学成首选。
```

#### 9.7.2 序列 reward / return

restoration 不是只做一步，因此还需要评价一个完整工具序列。

对于序列：

```text
τ = (s_0, a_0, s_1, a_1, ..., s_T)
```

序列 return 定义为：

```text
R_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T
```

其中：

```text
γ 是折扣因子，通常取 0.8-0.99。
```

设计动机：

```text
某个动作短期可能收益一般，但它可能为后续工具创造更好条件。
```

例如：

```text
先 decosmic_ray:
  单步 PSNR 提升可能不大；
  但它移除局部尖峰后，后续 dehazing / denoising 更稳定；
  整个序列最终 reward 很高。
```

因此 Q head 最好学习 return，而不是只学习单步 reward。

#### 9.7.3 Stop reward

Stop 也是一个动作。

如果在状态 `s_t` 停止：

```text
a_t = stop
I_final = I_t
```

stop reward 可以定义为：

```text
r_stop =
    α_final * OARS_t
  + α_down  * DownstreamScore_t
  - β_res   * ResidualDegradation_t
  - β_risk  * DamageRisk_t
  - β_time  * TotalRuntimeNorm
```

如果当前图像已经满足：

```text
退化残留低；
下游 proxy 稳定；
继续执行工具的收益预测很小；
损伤风险较高；
```

那么 stop 应该得到较高 reward。

如果当前仍然存在明显 dqg、noise 或 motion blur，但 Policy Net 选择 stop，则 stop reward 应该低。

#### 9.7.4 Rollback reward

Rollback 的作用是处理“工具造成副作用”的情况。

如果上一步从 `I_{t-1}` 得到 `I_t`，但出现：

```text
star_damage_risk 上升；
target_damage_risk 上升；
downstream proxy 下降；
GT-based reward 为负；
```

则 rollback 应该被标记为正样本。

离线可定义：

```text
r_rollback =
    max(0, -r_{t-1})
  + κ_star   * DamageStar
  + κ_target * DamageTarget
  - κ_time   * RollbackCost
```

也就是说，上一步越有害，rollback 的 reward 越高。

### 9.8 离线 LLM / Oracle 如何生成训练样本

Policy Net 训练基于离线 LLM / oracle rollout。

核心原则：

```text
LLM 积累 sample 时要看到 Policy Net 未来会用到的所有 online-available features。
LLM 离线阶段可以额外使用 GT-based privileged metrics 计算 reward。
Policy Net 训练输入不能包含 GT。
```

#### 9.8.1 LLM 离线阶段能看到什么

LLM / oracle 的输入分两部分。

第一部分是在线可见状态，也就是未来 Policy Net 也能看到的：

```text
degradation features
risk features
downstream proxy features
image statistics
history features
memory statistics
retrieved experience features，可选
```

第二部分是离线特权信息，只用于评价和选动作，不进入 Policy Net 输入：

```text
clean GT
star mask
target mask
degradation mask
restoration target
下游 GT label
真实 Star Preservation
真实 Target Recall
真实 OARS
真实 PSNR / SSIM
```

设计思想是：

```text
让老师批改时可以看答案；
但学生考试时只能看题目。
```

因此，训练样本中必须明确区分：

```text
online_state:
  Policy Net 训练和在线部署都可用。

privileged_eval:
  只用于离线计算 reward、选择最优动作和分析失败原因。
```

#### 9.8.2 LLM / oracle 的探索方式

离线探索可以分三种强度。

第一种：单步枚举。

```text
对当前状态 s_t：
  枚举所有合法动作 a ∈ A_legal；
  分别调用 executor；
  用 GT-based reward 计算 r(s_t, a)；
  选择 reward 最高的动作作为 teacher action。
```

适合生成 Q head 训练数据。

第二种：beam search 序列探索。

```text
从 I_0 出发；
每一步保留 top-B 个候选序列；
每个候选继续扩展合法动作；
用累计 return 排序；
得到最优或近似最优工具顺序。
```

适合学习双退化、三退化下的工具顺序。

第三种：LLM-guided search。

```text
LLM 根据退化组合、经验库和当前状态提出候选动作子集；
系统只对这些候选动作实际运行 executor；
再用 GT reward 选择最优动作。
```

适合降低离线探索成本。

推荐实践：

```text
1. 对小规模验证集做 exhaustive / beam search，得到高质量 oracle 样本。
2. 对大规模训练集做 LLM-guided search，降低计算开销。
3. 把两类样本统一写入 verified memory。
```

#### 9.8.3 LLM plan-and-execute with rollback

LLM teacher 阶段不应该只做单步 next-action 决策，而应该更接近当前 AgenticIR 的 `plan-and-execute` 流程。

也就是说，LLM 首先根据初始状态生成一个完整恢复顺序：

```text
initial_plan = [
  denoising@SwinIR,
  dehazing@DehazeFormer,
  stop
]
```

然后不是盲目执行完整 plan，而是每执行一步就重新感知当前图像：

```text
I_t -> executor(action_t) -> I_{t+1}

重新运行：
  Star-DepictQA / Star-DepictQA-Lite
  downstream proxy
  image statistics
  GT-based privileged evaluator，离线阶段可用

得到：
  new_state
  reward
  risk_delta
  downstream_delta
```

LLM 根据反馈决定：

```text
1. continue:
   当前步骤有效，继续执行原 plan 的下一步。

2. skip:
   某个退化已经被前一步间接改善，跳过原计划中的某一步。

3. rollback:
   上一步工具造成星点/目标损伤或 reward 为负，回退到上一图像。

4. replan:
   当前状态和初始预测不一致，重新生成后续 plan。

5. stop:
   退化已足够低，下游 proxy 稳定，继续恢复收益不足。
```

因此，LLM 阶段的真实轨迹不是固定开环计划，而是：

```text
plan -> execute one step -> evaluate -> continue / rollback / replan / stop
```

这个设计的动机是：

```text
1. restoration executor 的效果不确定。
2. 多退化之间会相互影响，一个工具可能间接改善或暴露另一类退化。
3. 过度执行原计划可能伤害弱星点和小目标。
4. rollback / replan 能显式生成失败样本，是训练 Policy Net 安全决策的重要来源。
```

离线 LLM plan-and-execute 可以记录两类标签。

第一类是全局计划标签：

```text
planned_sequence:
  初始 LLM 认为合理的完整恢复顺序。

final_executed_sequence:
  经过反馈、rollback、replan 后实际执行成功的顺序。

plan_edit_trace:
  哪些步骤被跳过、替换、回滚或提前停止。
```

第二类是逐步决策标签：

```text
state_t -> action_t
state_t -> tool_t
state_t -> continue / stop / rollback / replan
state_t -> Q_label(s_t, a)
```

Policy Net 最终主要学习第二类逐步标签，但第一类全局计划标签也有价值：

```text
1. 可用于训练 optional plan head，预测一个粗略恢复顺序。
2. 可用于经验库统计 best_historical_order_id。
3. 可用于初始化在线调度的 action prior。
4. 可用于解释：为什么当前步骤是整个恢复计划中的第一步。
```

因此，STAR-Agent 可以采用“两层规划”：

```text
LLM 离线阶段:
  full-plan first，再逐步执行、评估、rollback、replan。

Policy Net 在线阶段:
  默认输出 next action，但这个 next action 是从离线 full-plan 轨迹中蒸馏出来的。
  如需增强，也可以加入 plan head，输出短 horizon 的粗计划。
```

这和当前 AgenticIR 的思想是一致的：

```text
先有计划；
执行中检查；
出错可回滚；
必要时重规划。
```

区别在于 STAR-Agent 进一步把这套过程结构化为可训练样本，让轻量 Policy Net 能继承 LLM / AgenticIR 风格的调度经验。

#### 9.8.4 完整 transition 样本

```json
{
  "online_state_before": {
    "degradation": "...",
    "risk": "...",
    "downstream_proxy": "...",
    "history": "...",
    "memory_features": "..."
  },
  "llm_decision": {
    "action": "dehazing",
    "tool": "TinyStrayLightNet",
    "stop": false,
    "reason": "..."
  },
  "privileged_eval_after": {
    "gt_oars_after": 0.65,
    "gt_target_recall_after": 0.57,
    "gt_star_preservation_after": 0.96
  },
  "reward": 0.13,
  "online_state_after": "...",
  "done": false
}
```

训练时：

```text
输入: online_state_before
监督: action / tool / stop / reward
不输入: privileged_eval_after
```

#### 9.8.5 样本标签如何确定

对一个状态 `s_t`，如果离线枚举了多个动作：

```text
A_legal = {a_1, a_2, ..., a_K}
```

每个动作都有 reward 或 return：

```text
R(s_t, a_1), R(s_t, a_2), ..., R(s_t, a_K)
```

teacher action 可以定义为：

```text
a_teacher = argmax_a R(s_t, a)
```

如果最高 reward 仍然很低或为负，则说明继续恢复意义不大，此时：

```text
如果当前 residual_degradation_score 已低:
  teacher action = stop

如果上一动作造成明显损伤:
  teacher action = rollback

否则:
  teacher action = least harmful action 或 stop
```

为了避免训练过于硬，可以把 reward 转成 soft label：

```text
p_teacher(a | s_t) = softmax(R(s_t, a) / τ)
```

其中：

```text
τ 是温度系数。
τ 小时更接近 one-hot；
τ 大时保留多个可行动作的相对好坏。
```

这种 soft label 对 restoration agent 很有用，因为很多场景并不只有唯一正确顺序。

#### 9.8.6 LLM 和 Policy Net 的职责边界

离线 LLM / oracle 做的是：

```text
1. 读取当前 online_state。
2. 结合经验库和图像退化描述提出初始完整恢复计划。
3. 按 plan 执行一步，并在每一步后重新评估。
4. 根据反馈决定 continue、skip、rollback、replan 或 stop。
5. 在离线环境中调用多个 executor 或多个工具序列。
6. 借助 GT、mask 和下游标注计算 reward。
7. 选择最优或近似最优动作与最终执行序列。
8. 生成结构化训练样本和可解释 reason。
```

Policy Net 学的是：

```text
1. online_state -> action 的映射。
2. online_state -> tool 的映射。
3. online_state -> stop / rollback 的判断。
4. online_state -> Q value / expected reward 的估计。
```

Policy Net 不学习：

```text
1. 直接读取自然语言经验。
2. 在线访问 clean GT。
3. 在线枚举运行所有 executor。
4. 在线重新推理复杂工具说明。
```

这也是为什么经验库需要被结构化：

```text
LLM 可以读自然语言经验；
Policy Net 更适合读固定维度统计特征或检索后的结构化特征。
```

一个完整离线样本可以理解为：

```text
input_to_policy:
  online_state_before

teacher_from_llm_or_oracle:
  best_action
  best_tool
  stop_label
  rollback_label
  q_label_for_candidate_actions

privileged_only_for_reward:
  clean GT
  masks
  downstream GT
  OARS / target recall / star preservation
```

因此，LLM 不是在线大脑，而是离线数据生成器、策略探索器和解释器；Policy Net 才是最终部署时真正执行调度的轻量 planner。

### 9.9 Policy Net 训练策略

建议从简单到复杂：

#### 9.9.1 基础行为克隆

```text
L_action = CE(π_action(. | s_t), a_teacher)

L_tool = CE(π_tool(. | s_t), tool_teacher)
```

行为克隆的目标是让 Policy Net 模仿 LLM / oracle 的动作选择。

设计动机：

```text
先让 Policy Net 学会“老师一般会怎么做”，保证基础调度能力稳定。
```

缺点是：

```text
普通行为克隆只知道老师选了什么，不知道这个动作到底好多少。
```

因此需要 reward-weighted behavior cloning 和 Q head。

#### 9.9.2 Reward-weighted behavior cloning

普通 CE loss 可以加入 reward 权重：

```text
L_action = w_t * CE(π_action(. | s_t), a_teacher)

L_tool = w_t * CE(π_tool(. | s_t), tool_teacher)
```

其中：

```text
w_t = clip(normalize(max(R_teacher, 0)), w_min, w_max)
```

如果使用 soft label：

```text
L_action_soft = w_t * KL(p_teacher(. | s_t) || π_action(. | s_t))
```

这样做的含义不是“越像 LLM 权重越大”，而是：

```text
高 reward 的老师决策更值得重点学习；
低 reward 或不确定样本对训练影响更小；
负 reward 的动作不应被当作正样本强化。
```

例如：

```text
状态 A:
  denoise 后 OARS 大幅上升，弱目标没有损伤，R=0.25。
  这个样本权重大。

状态 B:
  denoise 和 dehaze 都差不多，R=0.02。
  这个样本权重小，避免模型过拟合偶然选择。
```

#### 9.9.3 Q head 与 Q 学习

Q head 的目标是：

```text
不只学习“老师选了哪个动作”，还学习“每个动作预期能带来多少收益”。
```

离线阶段可以对同一个状态执行多个候选动作。

对于状态 `s_t`：

```text
A_legal = {stop, denoise, dehaze, decosmic, dead_pixel_repair, desmear, deblur, rollback}
```

对每个动作：

```text
I_{t+1}^{(a)} = Tool_a(I_t)

R_label(s_t, a) = GT_based_return(s_t, a)
```

如果只看一步：

```text
Q_label(s_t, a) = r(s_t, a)
```

如果看后续序列：

```text
Q_label(s_t, a) = r_t + γ max_{a'} Q_label(s_{t+1}, a')
```

实际实现中可以用离线 rollout 的累计 return 作为 label：

```text
Q_label(s_t, a_t) = R_t
```

Q head 输出：

```text
Q_pred(s_t) = [Q_pred(s_t, a_1), ..., Q_pred(s_t, a_K)]
```

Q loss：

```text
L_q = mean_{a in A_observed} SmoothL1(Q_pred(s_t, a), Q_label(s_t, a))
```

其中：

```text
A_observed:
  离线实际执行过并有 reward label 的动作集合。
```

如果一个状态下没有枚举全部动作，只枚举了部分动作，就只对有标签的动作计算 `L_q`。

训练完成后，在线阶段没有 GT，也不会把所有工具都跑一遍。

在线只做：

```text
输入当前 state s_t；
Q head 直接预测每个动作的期望收益；
选择 Q 值高且满足安全约束的动作。
```

也就是：

```text
action = argmax legal_action Q(state, action)
```

这里要特别区分：

```text
离线:
  Q_label 来自 GT、mask、下游标签和真实工具执行结果。

在线:
  Q_pred 来自 Policy Net 预测，不需要 GT。
```

#### 9.9.4 Stop head 训练

Stop label 可以由 oracle 序列得到：

```text
如果当前状态继续任何动作的最大收益都低于阈值:
  stop_label = 1

如果当前图像已经达到任务指标要求:
  stop_label = 1

如果继续动作损伤风险高于收益:
  stop_label = 1

否则:
  stop_label = 0
```

stop loss：

```text
L_stop = BCE(stop_pred, stop_label)
```

在线 stop 不建议完全交给神经网络，而是结合规则：

```text
stop if:
  p_stop > threshold
  and max_a Q(s_t, a) < q_continue_threshold
  and residual_degradation_score < threshold
  and downstream_proxy is stable or improved
  and no high damage risk
```

强制停止条件：

```text
达到最大步数
超过时间/算力预算
连续两步 task gain <= 0
damage risk 明显上升且无可用修复动作
```

#### 9.9.5 Rollback head 训练

Rollback label 来自负效果样本。

如果上一步动作满足：

```text
r_{t-1} < negative_threshold
or DamageStar > threshold
or DamageTarget > threshold
or downstream GT metric 明显下降
```

则：

```text
rollback_label = 1
```

否则：

```text
rollback_label = 0
```

rollback loss：

```text
L_rollback = BCE(rollback_pred, rollback_label)
```

在线时 rollback 可以由三类信号共同触发：

```text
1. rollback_pred 高；
2. 上一步 proxy reward 为负；
3. risk delta 显示星点/目标损伤风险明显上升。
```

#### 9.9.6 Auxiliary head，可选

为了让 Policy Net 不只是记住动作，还能理解动作后果，可以增加辅助预测：

```text
pred_delta_oars
pred_delta_star_risk
pred_delta_target_risk
pred_delta_background_residual
pred_runtime
```

对应 loss：

```text
L_aux =
    MSE(pred_delta_oars, delta_oars_label)
  + MSE(pred_delta_star_risk, delta_star_risk_label)
  + MSE(pred_delta_target_risk, delta_target_risk_label)
  + MSE(pred_runtime, runtime_label)
```

这类 head 不一定在线直接用于决策，但能增强 backbone 对“工具后果”的理解。

#### 9.9.7 Multi-head joint loss

最终训练目标：

```text
L_total =
    λ_action   * L_action
  + λ_tool     * L_tool
  + λ_stop     * L_stop
  + λ_rollback * L_rollback
  + λ_q        * L_q
  + λ_aux      * L_aux
  + λ_reg      * ||θ||_2
```

推荐训练顺序：

```text
Stage 1: 只训练 action / tool / stop，先学会模仿 oracle。
Stage 2: 加入 reward-weighted BC，让高价值样本权重更高。
Stage 3: 加入 Q head，让模型学会给候选动作打分。
Stage 4: 加入 rollback 和 auxiliary head，提升安全性和解释性。
Stage 5: 用新 verified memory 做周期性微调。
```

#### 9.9.8 在线推理计算流程

在线第 `t` 步：

```text
Step 1: 输入当前图像 I_t。

Step 2: Star-DepictQA-Lite 输出 degradation features 和 risk features。

Step 3: 下游 proxy module 输出 star matching / target detection proxy。

Step 4: image statistics extractor 输出背景、亮点、局部对比度等统计量。

Step 5: memory module 查表或检索，得到 tool memory features。

Step 6: 拼接得到 state s_t。

Step 7: Policy Net 输出 action logits、tool logits、stop_prob、rollback_prob、Q values。
  这里的 action logits 不是最终动作，而是每个候选动作的原始偏好分数。
  Q values 也不是最终动作，而是每个候选动作的预期收益估计。

Step 8: decision layer 结合 action logits、Q values、经验先验、runtime 和风险约束，计算最终动作分数。

Step 9: 从合法候选动作中选择最终动作，执行 executor 或 stop / rollback。

Step 10: 重新感知新图像，更新 state 和 online_unverified_memory。
```

动作选择：

```text
Step 7 输出的是“候选动作分布”和“候选动作价值”；
Step 8 才是“最终动作选择规则”；
Step 9 才真正执行动作。
```

如果采用最简单版本，可以直接用 action head：

```text
a_t = argmax_a π_action(a | s_t)
```

但更推荐使用安全融合版本：

```text
score(a) =
    log π_action(a | s_t)
  + η * Q_norm(s_t, a)
  + ρ * memory_prior(a | s_t)
  - μ * runtime_cost(a)
  - ν * predicted_damage_risk(a)
```

这个公式的含义是：

```text
log π_action(a | s_t):
  Policy Net action head 认为这个动作像不像 LLM / oracle 会选的动作。

Q_norm(s_t, a):
  Q head 认为这个动作未来累计收益有多高。

memory_prior(a | s_t):
  经验库中类似状态下，这个动作或工具过去是否有效。

runtime_cost(a):
  执行动作需要付出的时间和算力代价。

predicted_damage_risk(a):
  这个动作可能伤害星点或目标的风险。
```

其中：

```text
memory_prior:
  来自经验库统计或 RAG 检索的工具先验。

runtime_cost:
  当前工具耗时或算力代价。

predicted_damage_risk:
  当前动作可能伤害星点/目标的风险。
```

最终：

```text
a_t = argmax_{a in legal actions} score(a)
```

举例：

```text
action head:
  denoising 概率最高，因为 LLM 经验中 noise high 通常先去噪。

Q head:
  dehazing 的预计收益也很高，因为当前 dqg/solar stray light 很明显。

memory prior:
  历史上 noise+dqg 组合里，denoise -> dehaze 的成功率最高。

risk penalty:
  当前弱星点很多，强去噪的 damage risk 偏高。

最终 decision layer:
  可能仍然选择 denoising，但会选择更保守的 denoising tool；
  或者如果 damage risk 太高，就先选择 dehazing 或 stop/rollback。
```

如果满足 stop 条件：

```text
输出 I_t，结束。
```

如果满足 rollback 条件：

```text
回退到 I_{t-1}；
把上一工具加入当前图像的临时禁用列表；
重新规划下一步动作。
```

#### 9.9.9 Policy Net 微调与安全更新

Policy Net 的微调不建议直接在星上实时进行，而建议采用“在线记录、地面复核、离线更新、再部署”的方式。

原因是：

```text
1. 在线没有 clean GT，无法确认 restoration 是否真的保护了弱星点和小目标。
2. 在线 proxy 可能被伪星点、伪目标或背景变化误导。
3. 航天系统需要稳定性和可追溯性，不适合边执行边大幅更新策略网络。
```

推荐微调流程：

```text
Step 1: 初始离线训练
  使用仿真数据、半仿真数据和 verified memory 训练初版 Policy Net。

Step 2: 在线部署
  Policy Net 只负责推理，同时记录 online_unverified_memory。

Step 3: 地面复核
  把在线日志回传地面，用强模型、人工检查、星表匹配、下游任务或可获得的 GT 做复核。

Step 4: 样本升级
  将可靠样本标记为 verified_positive / verified_negative。

Step 5: 增量微调
  用新增 verified memory 对 Policy Net 进行小学习率微调。

Step 6: 安全验证
  在固定验证集、极端退化集、真实数据集和下游任务集上评估。

Step 7: 再部署
  只有通过安全阈值后，才替换星上 Policy Net。
```

微调时建议混合三类数据：

```text
1. original_verified_memory:
  原始大规模离线样本，防止灾难性遗忘。

2. new_verified_memory:
  新回传、新复核的目标域样本，适应用真实分布。

3. hard_negative_memory:
  工具损伤星点、抹掉目标、产生伪星点、错误 stop 的失败样本。
```

微调 loss 可以写成：

```text
L_finetune =
    L_total_new
  + λ_replay * L_total_replay
  + λ_safe   * L_safe_negative
```

其中：

```text
L_total_new:
  新 verified memory 上的多 head loss。

L_total_replay:
  原始样本 replay loss，避免模型忘记旧场景。

L_safe_negative:
  对危险动作施加额外惩罚，例如在会损伤目标的状态下选择强去噪。
```

安全负样本可以定义为：

```text
如果某动作满足：
  DamageStar > threshold
  or DamageTarget > threshold
  or FalseTargetIncrease > threshold
  or downstream metric 明显下降

则该动作是 unsafe action。
```

对应安全 loss：

```text
L_safe_negative = - log(1 - π_action(unsafe_action | s_t))
```

也可以用 margin 约束：

```text
L_margin = max(0, margin + Q(s_t, unsafe_action) - Q(s_t, safe_action))
```

这样可以强制：

```text
Q(s_t, safe_action) > Q(s_t, unsafe_action)
```

这部分设计的目标不是让 Policy Net 更激进，而是让它更保守、更安全：

```text
宁可少做一步 restoration；
也不要为了提升背景指标而破坏星点和小目标。
```

### 9.10 经验统计如何获得

经验库不是 LLM 私有 memory，也不是 Policy Net 参数内部的隐式记忆，而是 STAR-Agent 的系统级共享 memory。它的作用是把历史 restoration rollout 中的“状态、动作、工具结果、下游反馈、失败原因”结构化保存下来，供离线蒸馏和在线调度共同使用。

推荐把经验库分成两层：

```text
Experience Memory
├── offline_verified_memory
│   └── 离线 LLM / exhaustive rollout / oracle evaluator 写入，带 GT 或强验证，可信度高
└── online_unverified_memory
    └── Policy Net 在线执行时写入，只包含 proxy 反馈和执行日志，需要后续复核
```

#### 9.10.1 谁可以写经验库

LLM / offline rollout 可以写入 verified experience：

```text
来源：
1. 全仿真数据，带 clean GT、star mask、target mask、degradation mask；
2. 半仿真数据，带 clean GT 和退化参数记录；
3. 离线穷举或半穷举多个 restoration 顺序；
4. 离线下游任务评价和 GT-based reward。
```

这类经验可以直接用于：

```text
训练 Policy Net；
统计 best order；
初始化 tool success rate；
提炼 scheduling skill；
分析失败案例。
```

Policy Net 在线运行时也可以写经验库，但它写入的是 unverified execution log：

```text
来源：
1. 在线 Star-DepictQA-Lite 输出；
2. 在线 downstream proxy 输出；
3. 工具执行成功/失败；
4. runtime；
5. 是否 rollback；
6. stop 前后的状态变化。
```

在线没有 clean GT，因此 Policy Net 不能直接写入真实 `PSNR`、真实 `Target Recall`、真实 `Star Preservation` 或真实 `OARS`。它只能写 proxy reward 和执行日志，并且必须标记为 `unverified` 或 `proxy_positive / proxy_negative`。

#### 9.10.2 离线 verified memory 的样本格式

离线阶段可以让 LLM / oracle planner 看到 GT 来评价不同恢复顺序，但最终保存给 Policy Net 学习的输入必须仍然是 online-available state。

示例：

```json
{
  "source": "offline_llm_rollout",
  "status": "verified_positive",
  "image_id": "frame_001",
  "combo": "noise_high+dqg_medium",
  "online_state_before": {
    "noise_prob": 0.96,
    "noise_level": 4,
    "dqg_prob": 0.88,
    "dqg_level": 3,
    "background_residual": 0.70,
    "star_damage_risk": 0.18,
    "target_damage_risk": 0.22,
    "plate_solve_success": 0,
    "matched_star_count": 23,
    "step_index": 0,
    "last_action_id": "none"
  },
  "action": {
    "subtask": "denoising",
    "tool": "TinyDenoise-Star"
  },
  "online_state_after": {
    "noise_prob": 0.12,
    "noise_level": 1,
    "dqg_prob": 0.86,
    "dqg_level": 3,
    "background_residual": 0.66,
    "star_damage_risk": 0.19,
    "target_damage_risk": 0.24,
    "plate_solve_success": 1,
    "matched_star_count": 51,
    "step_index": 1,
    "last_action_id": "denoising"
  },
  "privileged_eval": {
    "gt_oars_delta": 0.11,
    "gt_star_preservation": 0.96,
    "gt_target_recall": 0.92,
    "gt_background_residual_delta": 0.04
  },
  "reward": 0.09,
  "success": true,
  "failure_reason": null
}
```

注意这里有两个关键信息：

```text
online_state_before / online_state_after:
  Policy Net 训练和部署时都能看到的输入。

privileged_eval:
  只用于离线算 reward、筛选最优路径、标记 success/failure。
  不能作为 Policy Net 的直接输入。
```

因此，GT 在这里的角色是“老师批改作业的答案”，不是“学生考试时可见的提示”。

#### 9.10.3 在线 unverified memory 的样本格式

在线阶段，Policy Net 执行一次工具后写入：

```json
{
  "source": "online_policy_log",
  "status": "unverified",
  "image_id": "online_frame_1034",
  "combo_pred": "noise_high+dqg_medium",
  "state_before": {
    "noise_level": 4,
    "dqg_level": 3,
    "background_residual": 0.70,
    "star_damage_risk": 0.21,
    "target_damage_risk": 0.25,
    "plate_solve_success": 0,
    "matched_star_count": 23
  },
  "action": {
    "subtask": "denoising",
    "tool": "TinyDenoise-Star"
  },
  "state_after": {
    "noise_level": 1,
    "dqg_level": 3,
    "background_residual": 0.67,
    "star_damage_risk": 0.22,
    "target_damage_risk": 0.26,
    "plate_solve_success": 1,
    "matched_star_count": 51
  },
  "proxy_reward": 0.12,
  "runtime": 0.43,
  "tool_error": null,
  "rollback": false,
  "stop_after_action": false
}
```

这类样本可以立即用于：

```text
统计工具是否频繁报错；
统计某类图像上工具平均耗时；
发现部署域偏移；
给后续地面复核和再训练提供候选样本。
```

但它不能直接作为 verified positive 样本训练 Policy Net，因为在线 proxy 可能漏掉弱目标损伤或伪星点问题。

#### 9.10.4 在线日志如何升级为 verified memory

推荐闭环：

```text
Step 1: 离线 rollout 初始化 offline_verified_memory；
Step 2: Policy Net 在线执行，持续写 online_unverified_memory；
Step 3: 在线日志回传地面；
Step 4: 地面使用 clean GT、人工复查、强模型、下游任务或星表匹配进行复核；
Step 5: 通过复核的样本升级为 verified_positive / verified_negative；
Step 6: 使用新增 verified memory 重新训练或微调 Policy Net。
```

经验样本可以设置状态：

```text
unverified:
  在线刚记录，尚未复核。

proxy_positive:
  在线 proxy 指标显示有效，但没有 GT 确认。

proxy_negative:
  在线 proxy 指标显示变差或工具失败。

verified_positive:
  离线 GT / 人工 / 强模型确认该动作有效。

verified_negative:
  离线 GT / 人工 / 强模型确认该动作有害或无效。
```

星上部署时，建议在线只写日志和更新安全统计，不直接在线训练 Policy Net。这样更稳定，也更符合航天系统的安全约束。

#### 9.10.5 经验统计如何聚合

对 verified memory 和必要的 proxy memory 按照 `degradation combo + severity bin + tool/action` 聚合：

```text
tool_success_rate_current_combo
tool_avg_reward_current_combo
tool_star_damage_rate_current_combo
tool_target_gain_current_combo
tool_background_gain_current_combo
tool_runtime_mean
tool_failure_rate
best_historical_order_id
```

具体计算：

```text
tool_success_rate = success_count / total_count

tool_avg_reward = mean(reward)

tool_star_damage_rate =
  mean(star_damage_delta > threshold)

tool_target_gain =
  mean(target_recall_delta or target_proxy_delta)

tool_background_gain =
  mean(background_residual_before - background_residual_after)

tool_failure_rate =
  failure_count / total_count

best_historical_order_id =
  argmax mean(final_reward over action_sequence)
```

success 的定义要区分离线和在线：

```text
offline success:
  GT reward > 0
  and target / star preservation not worse
  and degradation residual reduced
  and downstream GT metric not worse

online proxy success:
  proxy_reward > 0
  and Star-DepictQA-Lite degradation score reduced
  and downstream proxy not worse
  and risk score below threshold
  and tool did not fail
```

#### 9.10.6 Policy Net 如何使用经验库

不启用 RAG 时，经验库先被压缩为固定维度统计特征：

```text
当前 state:
  noise_high + dqg_medium

memory lookup:
  denoising_success_rate = 0.86
  stray_first_success_rate = 0.61
  denoising_avg_reward = 0.10
  stray_first_target_damage_rate = 0.23
  best_order_id = denoising_then_stray
```

然后拼接进 Policy Net 输入：

```text
state = [
  degradation features,
  risk features,
  downstream proxy features,
  history features,
  memory statistics features
]
```

启用结构化 RAG 时，经验库先检索相似历史案例：

```text
case 1: noise_l4+dqg_l3, denoise -> stray, final_reward +0.24
case 2: noise_l5+dqg_l2, denoise -> stray, final_reward +0.21
case 3: noise_l4+dqg_l4, stray first, target_damage high
```

再把 top-k 案例压缩为：

```text
retrieved_best_action_id
retrieved_best_order_id
retrieved_avg_reward
retrieved_failure_rate
retrieved_topk_action_histogram
retrieved_similarity_mean
```

最后输入 Policy Net。

一句话总结：

```text
LLM / offline rollout 写入有 GT 验证的高质量经验；
Policy Net 在线写入无 GT 的执行日志和 proxy 反馈；
经验库经过地面复核后不断升级；
Policy Net 使用的是经验库的统计特征或结构化检索特征，而不是直接读取自然语言经验。
```

---

## 10. 调度流程

### 10.1 输入

输入一张空间图像：

```text
I_0
```

初始化：

```text
current_image = I_0
history = []
step = 0
```

### 10.2 初始状态感知

运行：

```text
Star-DepictQA-Lite(current_image)
下游 proxy module, LOST/tetra3rs / detector
image statistics extractor
memory retriever / tool statistics lookup
```

得到：

```text
degradation levels
observation risk
downstream proxy
image stats
memory features
```

构建：

```text
state_0
```

### 10.3 Policy Net 决策

输入：

```text
state_t
```

输出：

```text
next_action
selected_tool
stop_prob
rollback_prob
expected_reward
```

若 stop：

```text
输出 current_image
```

否则执行 selected_tool。

### 10.4 Executor 执行

```text
current_image -> selected_executor -> candidate_image
```

记录：

```text
action
tool
runtime
```

### 10.5 执行后反馈

对 candidate_image 重新运行：

```text
Star-DepictQA-Lite
下游 proxy module
image statistics
```

计算：

```text
delta_degradation
delta_risk
delta_downstream_proxy
estimated_task_gain
```

更新：

```text
history
memory log
state_{t+1}
```

### 10.6 Stop / Continue / Rollback

Policy Net 根据新 state 决定：

```text
stop: 输出 candidate_image
continue: 进入下一步
rollback: 回到上一张图，并禁止或降低 last_tool 置信度
```

### 10.7 输出

最终输出：

```text
restored image
execution trace
selected tools
step-wise degradation/risk/proxy changes
final confidence
runtime cost
```

这使系统不仅输出图像，也输出可解释恢复过程。

---

## 11. 参考链接

公开工具和方法参考：

```text
GraXpert: https://github.com/Steffenhir/GraXpert/
BSC-Net paper: https://www.mdpi.com/2072-4292/14/19/4852
ccdproc cosmicray_lacosmic: https://ccdproc.readthedocs.io/en/2.5.1/api/ccdproc.cosmicray_lacosmic.html
Astropy bad pixel mask guide: https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/08-02-Creating-a-mask.html
Siril cosmetic correction: https://siril.readthedocs.io/en/latest/processing/cc.html
Dynamic star map degradation model: https://www.mdpi.com/2304-6732/9/10/673
Star tracker motion blur simulation: https://opg.optica.org/josab/abstract.cfm?uri=josab-39-11-2934
FY-3C stray light simulation: https://www.mdpi.com/2072-4292/13/24/5037
LIME lunar irradiance model: https://lime.uva.es/
Ansys off-axis moon stray-light analysis example: https://optics.ansys.com/hc/en-us/articles/43071106491027-How-to-perform-stray-light-analysis
TESSCut MAST: https://mast.stsci.edu/tesscut/docs/index.html
Kepler MAST: https://archive.stsci.edu/kepler/
LOST: https://github.com/UWCubeSat/lost
tetra3rs: https://tetra3rs.dev/tutorials/basic-solve/
Astrometry.net: https://astrometry.net/doc/
ASTRiDE: https://github.com/dwkim78/ASTRiDE
```
