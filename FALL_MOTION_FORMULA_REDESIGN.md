# 跌倒判定公式与二维指标设计

## 1. 目的

本文整理当前项目中已有的坐标、重心、速度、分类告警公式，并给出一套可落地的二维物理指标方案。

核心目标是把跌倒判定从单一分类输出改为：

```text
模型分类概率 + 姿态几何指标 + 下坠动力学指标 + 时间确认逻辑
```

其中“二维指标”指：

- 姿态几何维度 `G_t`：人体是否从竖直支撑状态转为低姿态、横向姿态。
- 下坠动力学维度 `D_t`：人体重心是否出现明显向下速度或向下加速度。

## 2. 原来的公式

### 2.1 二维训练数据坐标归一化

`build_coco17_xy60_dataset.py` 当前生成二维输入时，只保留 `x,y` 两个通道，丢弃 score。有效关节的坐标按画面宽高归一化：

$$
x'_{t,j} = \frac{x_{t,j}}{W / 2} - 1
$$

$$
y'_{t,j} = \frac{y_{t,j}}{H / 2} - 1
$$

其中：

- `W` 为视频宽度。
- `H` 为视频高度。
- `j` 为 COCO17 关节编号。
- 无效关节坐标置为 `0`。

该公式得到的是画面坐标系中的绝对位置归一化坐标，不是人体局部坐标。

### 2.2 三通道数据公式

`build_coco17_walk_replacement_dataset.py` 生成三通道输入：

$$
X_{t,j} = (x_{t,j}, y_{t,j}, q_{t,j})
$$

其中：

- `x,y` 保留原始像素坐标。
- `q` 为关键点置信度或由 NTU tracking state 近似得到的 score。

该路径没有执行画面宽高归一化，也没有执行 Root Motion 归一化。

### 2.3 Root Motion 局部坐标公式

`ROOT_MOTION_LOGIC.md` 和 `visualize_relative_coordinates.py` 中已有一套可视化用 Root Motion 逻辑。

Root 横坐标优先使用左右髋中心：

$$
O_{x,t} =
\frac{x_{\text{leftHip},t} + x_{\text{rightHip},t}}{2}
$$

Root 纵坐标优先使用两个脚踝中更靠近画面底部的点：

$$
O_{y,t} =
\max(y_{\text{leftAnkle},t}, y_{\text{rightAnkle},t})
$$

原始 Root 点为：

$$
O_t = (O_{x,t}, O_{y,t})
$$

Root 点经过 5 帧中值滤波：

$$
\bar O_t =
\operatorname{Median}(O_{t-2}, O_{t-1}, O_t, O_{t+1}, O_{t+2})
$$

骨长尺度参考值：

$$
L_b^{ref} =
\operatorname{Median}_t(L_{t,b})
$$

每帧尺度倍率：

$$
r_{t,b} =
\frac{L_{t,b}}{L_b^{ref}}
$$

$$
s_t =
\operatorname{Median}_b(r_{t,b})
$$

最终归一化尺度：

$$
S_t = L_{\text{canonical}} \cdot s_t
$$

局部相对坐标：

$$
P'_{t,j} =
\frac{P_{t,j} - \bar O_t}{S_t}
$$

### 2.4 局部人体重心公式

按身体分段质量加权计算局部重心：

$$
C_t =
\frac{\sum_b m_b C_{t,b}}
{\sum_b m_b}
$$

其中：

- `b` 为身体分段。
- `m_b` 为身体分段质量比例。
- `C_{t,b}` 为分段中心。

当身体分段不足时，回退到有效关节均值或 Root 点。

### 2.5 局部重心速度公式

使用中心差分计算局部重心速度：

$$
V_t =
\frac{C_{t+1} - C_{t-1}}
{(f_{t+1} - f_{t-1}) / FPS}
$$

连续帧情况下：

$$
V_t =
\frac{C_{t+1} - C_{t-1}}
{2\Delta t}
$$

速度单位为：

```text
归一化骨长 / 秒
```

### 2.6 当前 softmax 告警公式

`fall_detection.classify_probabilities()` 当前使用 CTR-GCN 输出概率：

$$
p_k =
\frac{e^{z_k}}
{\sum_i e^{z_i}}
$$

Top-1 类别：

$$
k^* = \arg\max_k p_k
$$

跌倒分数：

$$
S_{\text{fall}} = p_{k_{\text{fall}}}
$$

告警分组：

$$
\text{group} = G(k^*)
$$

其中 `G(k)` 将类别映射为：

```text
normal / fall-like / fall
```

当前外部告警直接等于 top-1 类别所在分组：

$$
\text{external\_alarm} = G(k^*)
$$

## 3. 当前公式的问题

### 3.1 坐标空间不一致

当前项目里至少存在三套坐标空间：

- 三通道训练数据：原始像素 `x,y,score`。
- 二通道 `xy60` 数据：画面宽高归一化 `x',y'`。
- Root Motion 可视化：人体局部坐标 `P'`。

如果训练使用一种坐标空间，推理使用另一种坐标空间，模型输入分布会偏移。即使网络结构不变，分类概率也会不稳定。

### 3.2 画面归一化仍然保留相机和位置偏差

二维公式：

$$
x' = \frac{x}{W/2} - 1,\quad
y' = \frac{y}{H/2} - 1
$$

只能消除视频分辨率差异，不能消除：

- 人在画面左侧、右侧、远处、近处造成的分布差异。
- 摄像机俯仰、安装高度带来的坐标偏差。
- 行走、跑步时人体整体平移对模型的干扰。

### 3.3 Root Motion 会移除部分跌倒线索

Root Motion 局部坐标适合学习人体姿态，但它主动消除了人物在画面中的整体位移：

$$
P'_{t,j} =
\frac{P_{t,j} - \bar O_t}{S_t}
$$

这会削弱“整个人向地面快速移动”的全局线索。跌倒检测不能只依赖局部姿态，还需要保留重心下坠的动力学指标。

### 3.4 像素级速度和加速度不可直接跨视频比较

`visualize_com_velocity.py` 和 `visualize_com_acceleration.py` 中的速度单位分别是：

```text
px/s
px/s^2
```

像素速度受分辨率、人物距离、焦距影响。相同真实运动在不同画面尺度下会得到不同数值，因此不能作为最终阈值的直接依据。

### 3.5 当前告警只依赖 top-1 分组

当前逻辑：

$$
\text{external\_alarm} = G(\arg\max_k p_k)
$$

主要问题：

- `fall_score = p_{fall}` 只是展示值，不直接决定是否告警。
- fall-like 类别概率和 fall 类别概率没有聚合成风险分数。
- 单帧或单窗口 top-1 容易抖动。
- 没有用人体几何和下坠运动约束误报。
- running、squat down、lying、sitting 等动作容易和 fall/fall-like 混淆。

## 4. 建议的二维指标

### 4.1 二维指标定义

建议把物理判定拆成两个互补维度：

```text
G_t: 姿态几何分数，回答“人现在像不像倒下/低姿态？”
D_t: 下坠动力学分数，回答“刚才是否发生明显向下坠落？”
```

二维指标不替代 CTR-GCN，而是作为分类概率的校验和融合项。

### 4.2 姿态几何分数 `G_t`

姿态几何分数由三个子指标组成：

```text
G_h: 身体高度压缩
G_theta: 躯干横向化
G_c: 重心接近地面
```

#### 身体高度压缩

在局部坐标中，取有效身体关节的纵向跨度：

$$
H_t =
\max_{j \in \mathcal V_t} P'_{t,j,y}
-
\min_{j \in \mathcal V_t} P'_{t,j,y}
$$

用历史站立高度或轨迹高分位作为参考：

$$
H^{ref}_t =
\operatorname{P90}_{\tau \in [t-K,t]}(H_\tau)
$$

高度压缩分数：

$$
G_h(t) =
\operatorname{clip}
\left(
1 - \frac{H_t}{H^{ref}_t + \epsilon},
0,
1
\right)
$$

#### 躯干横向化

肩中心与髋中心分别为：

$$
S_t =
\frac{P'_{t,\text{leftShoulder}} + P'_{t,\text{rightShoulder}}}{2}
$$

$$
H_t^{hip} =
\frac{P'_{t,\text{leftHip}} + P'_{t,\text{rightHip}}}{2}
$$

躯干轴向量：

$$
U_t = S_t - H_t^{hip}
$$

横向化分数：

$$
G_\theta(t) =
\frac{|U_{t,x}|}
{\lVert U_t \rVert_2 + \epsilon}
$$

站立时 `G_theta` 接近 `0`，横躺时接近 `1`。

#### 重心接近地面

局部坐标中 Root 原点接近脚下，图像 `y` 轴向下。站立时人体重心通常在 Root 上方，即 `C_{t,y}` 更小；倒下后重心接近地面，即 `C_{t,y}` 增大。

历史站立重心参考：

$$
C_y^{ref}(t) =
\operatorname{P10}_{\tau \in [t-K,t]}(C_{\tau,y})
$$

重心下移分数：

$$
G_c(t) =
\operatorname{clip}
\left(
\frac{C_{t,y} - C_y^{ref}(t)}
\Delta C_y + \epsilon},
0,
1
\right)
$$

其中 `Delta C_y` 可取训练集统计得到的站立到倒地重心差，或先用经验值再用验证集校准。

#### 姿态几何总分

$$
G_t =
w_h G_h(t)
+ w_\theta G_\theta(t)
+ w_c G_c(t)
$$

约束：

$$
w_h + w_\theta + w_c = 1
$$

建议初始权重：

```text
w_h = 0.35
w_theta = 0.35
w_c = 0.30
```

### 4.3 下坠动力学分数 `D_t`

局部重心速度：

$$
V_t =
\frac{C_{t+1} - C_{t-1}}
{(f_{t+1} - f_{t-1}) / FPS}
$$

局部重心加速度：

$$
A_t =
\frac{V_{t+1} - V_{t-1}}
{(f_{t+1} - f_{t-1}) / FPS}
$$

向下速度分量：

$$
v^{down}_t = \max(V_{t,y}, 0)
$$

向下加速度分量：

$$
a^{down}_t = \max(A_{t,y}, 0)
$$

归一化下坠速度分数：

$$
D_v(t) =
\operatorname{clip}
\left(
\frac{v^{down}_t}{\theta_v},
0,
1
\right)
$$

归一化下坠加速度分数：

$$
D_a(t) =
\operatorname{clip}
\left(
\frac{a^{down}_t}{\theta_a},
0,
1
\right)
$$

动力学总分：

$$
D_t =
w_v D_v(t)
+ w_a D_a(t)
$$

约束：

$$
w_v + w_a = 1
$$

建议初始权重：

```text
w_v = 0.65
w_a = 0.35
```

`theta_v` 和 `theta_a` 不建议手工固定，应从验证集统计：

```text
theta_v = P75(v_down | fall windows)
theta_a = P75(a_down | fall windows)
```

## 5. 模块设计

### 5.1 预处理模块 `PoseNormalizer`

输入：

```text
track_id, frame_id, keypoints[17,2], scores[17], bbox, fps
```

输出：

```text
relative_keypoints[17,2]
valid_mask[17]
root_origin[2]
normalization_scale
```

职责：

- 关键点有效性判断。
- Root 原点估计。
- Root 中值滤波。
- 多骨长尺度估计。
- 输出训练和推理完全一致的局部坐标。

### 5.2 重心模块 `CenterOfMassEstimator`

输入：

```text
relative_keypoints[17,2], valid_mask[17], scores[17]
```

输出：

```text
local_com[2]
com_source
```

职责：

- 按身体分段质量比例估计重心。
- 缺失关节时按可用分段重新归一化。
- 无可用分段时回退到有效关节均值或 Root。

### 5.3 运动模块 `MotionFeatureExtractor`

输入：

```text
local_com sequence, frame_id sequence, fps
```

输出：

```text
local_velocity[2]
local_acceleration[2]
down_velocity
down_acceleration
D_t
```

职责：

- 用中心差分计算速度。
- 用中心差分计算加速度。
- 避免跨越大缺帧计算速度和加速度。
- 输出归一化的下坠动力学分数。

### 5.4 姿态模块 `GeometryFeatureExtractor`

输入：

```text
relative_keypoints[17,2], valid_mask[17], local_com[2], track history
```

输出：

```text
height_collapse
torso_horizontal
com_ground_proximity
G_t
```

职责：

- 计算身体高度压缩。
- 计算躯干横向化。
- 计算重心接近地面程度。
- 用滑动历史维护每条 track 的参考高度和参考重心。

### 5.5 融合模块 `FallRiskFusion`

输入：

```text
CTR-GCN probabilities
G_t
D_t
track temporal history
```

输出：

```text
fall_risk
internal_state
external_alarm
```

职责：

- 聚合模型 fall/fall-like 概率。
- 融合二维物理指标。
- 执行时间确认，降低单窗口抖动。

## 6. 修正版公式

### 6.1 模型概率聚合

保留 softmax：

$$
p_k =
\frac{e^{z_k}}
{\sum_i e^{z_i}}
$$

fall 类别概率：

$$
P_{\text{fall}} = p_{k_{\text{fall}}}
$$

fall-like 聚合概率：

$$
P_{\text{fall-like}} =
\sum_{k \in \mathcal K_{\text{fall-like}}} p_k
$$

模型风险分数：

$$
M_t =
\operatorname{clip}
\left(
P_{\text{fall}}
+ \lambda_{\text{like}} P_{\text{fall-like}},
0,
1
\right)
$$

建议初始值：

```text
lambda_like = 0.35
```

### 6.2 二维物理分数

姿态几何分数：

$$
G_t =
0.35G_h(t)
+ 0.35G_\theta(t)
+ 0.30G_c(t)
$$

下坠动力学分数：

$$
D_t =
0.65D_v(t)
+ 0.35D_a(t)
$$

### 6.3 物理跌倒分数

跌倒通常需要“姿态已经低/横向化”并且“最近出现过下坠”。因此建议用几何分数作为主门控，用动力学分数增强：

$$
R^{phys}_t =
G_t \cdot
\left(
\beta + (1-\beta)D_t
\right)
$$

其中：

```text
beta = 0.40
```

含义：

- `G_t` 高、`D_t` 高：快速倒下，高风险。
- `G_t` 高、`D_t` 低：可能是已倒地、躺下或慢速蹲下，中等风险。
- `G_t` 低、`D_t` 高：可能是跑跳、镜头抖动或追踪抖动，不应直接告警。
- `G_t` 低、`D_t` 低：正常。

### 6.4 最终融合风险分数

将模型风险和物理风险融合：

$$
R_t =
\alpha M_t
+ (1-\alpha)R^{phys}_t
$$

建议初始值：

```text
alpha = 0.60
```

如果当前模型在目标视频域上误报较多，可以降低 `alpha`，提高物理约束权重。

### 6.5 时间确认公式

单窗口风险不直接输出最终告警。建议使用滑动窗口聚合：

$$
\bar R_t =
\max_{\tau \in [t-N+1,t]} R_\tau
$$

或使用均值：

$$
\bar R_t =
\frac{1}{N}
\sum_{\tau=t-N+1}^{t} R_\tau
$$

建议优先使用最大值做召回，配合连续帧确认做防抖：

$$
\text{alarm}_t =
\mathbb 1
\left[
\sum_{\tau=t-N+1}^{t}
\mathbb 1(R_\tau \ge \theta_R)
\ge n_{\min}
\right]
$$

建议初始参数：

```text
N = 5
n_min = 2
theta_R = 0.65
```

### 6.6 状态机公式

建议输出三个状态：

```text
normal
fall-like
fall
```

状态规则：

$$
\text{state}_t =
\begin{cases}
\text{fall}, & \bar R_t \ge \theta_{\text{fall}} \land G_t \ge \theta_G \\
\text{fall-like}, & \bar R_t \ge \theta_{\text{like}} \lor M_t \ge \theta_M \\
\text{normal}, & \text{otherwise}
\end{cases}
$$

建议初始参数：

```text
theta_fall = 0.65
theta_like = 0.45
theta_G = 0.50
theta_M = 0.55
```

外部报警只对 `fall` 触发：

$$
\text{external\_alarm}_t =
\mathbb 1(\text{state}_t = \text{fall})
$$

`fall-like` 用于界面提示和内部监控，不直接触发强告警。

## 7. 推荐落地顺序

1. 先把训练和推理统一到同一套 `PoseNormalizer` 坐标空间。
2. 离线导出每条 track 的 `G_t, D_t, M_t, R_t`，用验证集看 running、walking、squat、fall 的分布。
3. 用验证集标定 `theta_v, theta_a, theta_R, theta_G`。
4. 再决定是否把 `G_t, D_t` 加入模型输入通道，或只作为后处理融合。

## 8. 最小实现建议

短期不改模型结构时，推荐只做后处理：

```text
YOLO Pose
→ tracking
→ CTR-GCN probabilities
→ PoseNormalizer
→ CenterOfMassEstimator
→ GeometryFeatureExtractor
→ MotionFeatureExtractor
→ FallRiskFusion
→ alarm
```

这样可以在不重新训练 CTR-GCN 的情况下，先验证二维指标是否能降低误报。

中期如果要重新训练，建议把输入统一为：

```text
channel 0: relative_x
channel 1: relative_y
channel 2: keypoint_score
```

并把每个窗口的辅助指标保存到元数据：

```text
G_mean, G_max, D_mean, D_max, R_phys_max
```

这些指标可用于分析、采样加权、困难负样本挖掘，也可以作为后续多分支模型的额外输入。
