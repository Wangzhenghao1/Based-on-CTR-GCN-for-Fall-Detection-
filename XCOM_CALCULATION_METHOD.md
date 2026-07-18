# 二维表观 XCoM 计算方法

本文档总结当前确定的 XCoM（Extrapolated Center of Mass，外推质心）计算流程。该方法面向 `COCO17` 二维骨架，依次完成骨段建模、质量加权人体质心、骨长尺度归一化、质心速度估计和二维表观 XCoM 计算。

完整流程为：

```text
COCO17 关节点
    -> 骨段与骨段质心
    -> 质量加权人体 CoM
    -> 骨长参考尺度与逐帧尺度
    -> 相对坐标和尺度归一化 CoM
    -> 尺度归一化 CoM 速度
    -> 2D apparent XCoM
```

## 1. 输入骨架

第 $t$ 帧、第 $i$ 个 COCO17 关节点为：

$$
\mathbf p_{t,i}^{abs}
=
\begin{bmatrix}
x_{t,i} \\
y_{t,i}
\end{bmatrix},
\qquad
s_{t,i}\in[0,1]
$$

有效关节点掩码定义为：

$$
M_{t,i}
=
\mathbb I\left(
s_{t,i}>\tau
\land
\mathbf p_{t,i}^{abs}\text{ 有效且非零}
\right)
$$

其中：

| 符号 | 含义 |
|---|---|
| $t$ | 时间帧索引，$t=0,\ldots,T-1$ |
| $T$ | 输入序列帧数，当前 CTR-GCN 输入为 64 |
| $i$ | COCO17 关节点编号 |
| $\mathbf p_{t,i}^{abs}$ | 绝对二维关节点坐标 |
| $s_{t,i}$ | 关节点置信度 |
| $\tau$ | 关键点有效性阈值 |
| $M_{t,i}$ | 关节点有效掩码 |

## 2. 骨段与骨段质心

第 $b$ 个骨段由近端关节 $i_b$ 和远端关节 $j_b$ 组成。

骨向量：

$$
\mathbf l_{t,b}
=
\mathbf p_{t,j_b}^{abs}
-
\mathbf p_{t,i_b}^{abs}
$$

骨长：

$$
L_{t,b}
=
\left\|\mathbf l_{t,b}\right\|_2
$$

骨段质心：

$$
\mathbf C_{t,b}^{abs}
=
\mathbf p_{t,i_b}^{abs}
+
\kappa_b
\left(
\mathbf p_{t,j_b}^{abs}
-
\mathbf p_{t,i_b}^{abs}
\right)
$$

$\kappa_b$ 表示骨段质心从近端到远端的位置比例。将其设为可学习参数时，使用以下约束：

$$
\kappa_b=\sigma(\theta_{\kappa_b})\in(0,1)
$$

$\kappa_b=0.5$ 表示骨段中点。更准确的实现可以使用人体测量学比例初始化，再由模型进行小范围修正。

## 3. 骨段质量分布

质量比例采用人体测量学先验初始化，并学习受约束的残差：

$$
w_b
=
\operatorname{softmax}
\left(
\log w_b^{prior}
+
\rho\tanh\delta_b
\right)
$$

因此：

$$
w_b>0,
\qquad
\sum_{b=1}^{B}w_b=1
$$

当前建议的男女平均质量比例为：

| 身体部分 | 质量比例 |
|---|---:|
| 头部 | 0.0681 |
| 躯干 | 0.4302 |
| 单侧上臂 | 0.0263 |
| 单侧前臂与手 | 0.02085 |
| 单侧大腿 | 0.1447 |
| 单侧小腿与脚 | 0.0590 |

左右侧分别计算，全部质量比例之和为 1。$\delta_b$ 初始化为 0，$\rho$ 建议设为 $0.1\sim0.2$，用于限制模型偏离质量先验的幅度。

人体绝对 CoM 为：

$$
\boxed{
\mathbf C_t^{abs}
=
\sum_{b=1}^{B}
w_b\mathbf C_{t,b}^{abs}
}
$$

## 4. 每条尺度连杆的二维参考长度

当前 `rel_coord_utils.py` 使用以下 12 条尺度连杆：

$$
\mathcal B_s=
\left\{
(5,6),(11,12),(5,11),(6,12),
(5,7),(7,9),(6,8),(8,10),
(11,13),(13,15),(12,14),(14,16)
\right\}
$$

它们覆盖肩宽、髋宽、躯干、双臂和双腿。这里称为“尺度连杆”，因为肩宽和髋宽不是严格意义上的人体骨骼。

对于尺度连杆 $b$，收集整段视频中的有效二维投影长度：

$$
\mathcal L_b
=
\left\{
L_{t,b}
\mid
M_{t,i_b}=1,
M_{t,j_b}=1
\right\}
$$

该连杆在这段视频中的二维参考长度为跨帧中位数：

$$
\boxed{
L_b^{proj,ref}
=
\operatorname{median}(\mathcal L_b)
}
$$

这里的 $L_b^{proj,ref}$ 不是人体真实三维骨长。更准确的二维投影关系为：

$$
L_{t,b}^{2D}
=
z_tq_{t,b}L_b^{true}
+
\varepsilon_{t,b}
$$

其中 $L_b^{true}$ 是真实骨长，$z_t$ 是人物远近产生的整体表观缩放，$q_{t,b}$ 是骨段朝向摄像机造成的投影缩短比例，$\varepsilon_{t,b}$ 是关键点检测误差。因此：

$$
L_b^{proj,ref}
\approx
\operatorname{median}_t(z_tq_{t,b})L_b^{true}
$$

$L_b^{proj,ref}$ 的准确含义是：该骨段在这段视频的常规人物距离和常规姿态下的二维投影参考长度。使用时间中位数可以减弱关节点抖动、遮挡和少量异常帧的影响。

## 5. 序列标准骨长

每条连杆都有自己的 $L_b^{proj,ref}$，但所有关节点必须使用同一个尺度进行等比例缩放，否则分别归一化每条骨段会破坏人体比例。因此，将所有有效参考长度汇总成一个序列统一标尺：

$$
\boxed{
S_0
=
\operatorname{median}_{b\in\mathcal B_s}
L_b^{proj,ref}
}
$$

$S_0$ 表示整段视频的人体基准表观骨架尺度。它不是人体身高，也不是某一根真实骨长，而是多条参考连杆的稳健代表值。使用中位数而不是平均数，是为了避免少量异常长或异常短的连杆主导整个尺度。

如果没有足够的有效尺度连杆，则退化为有效关节点包围框高度的时间中位数：

$$
S_0
=
\operatorname{median}_t
\left(
y_t^{max}-y_t^{min}
\right)
$$

## 6. 每帧表观缩放比例与尺度

第 $t$ 帧中，第 $b$ 条连杆相对自身参考长度的比例为：

$$
r_{t,b}
=
\frac{L_{t,b}}
{L_b^{proj,ref}+\epsilon}
$$

这个比例先消除了不同连杆本身长短的差异。理想弱透视条件下：

$$
r_{t,b}
\approx
\frac{z_t}{z_0}
$$

它近似描述第 $t$ 帧相对于整段视频基准状态的表观透视缩放比例。

当一帧至少存在 4 条有效尺度连杆时：

$$
\boxed{
r_t^{raw}
=
\operatorname{median}_{b\in\mathcal B_t}
r_{t,b}
}
$$

其中 $\mathcal B_t$ 为第 $t$ 帧的有效尺度连杆集合。使用多条连杆的中位数，相当于让所有有效连杆共同投票估计当前帧的整体缩放，避免依赖某一根容易被遮挡或透视缩短的骨段。

使用 7 帧窗口做时间中位数平滑：

$$
r_t^{smooth}
=
\operatorname{median}
\left\{
r_u^{raw}\mid |u-t|\le 3
\right\}
$$

将异常比例截断到合理范围：

$$
\hat r_t
=
\operatorname{clip}
\left(
r_t^{smooth},0.5,2.0
\right)
$$

经过时间平滑和截断后的 $\hat r_t$，表示当前帧相对于序列基准状态的无量纲整体表观缩放比例。

最终逐帧表观骨架尺度为：

$$
\boxed{
S_t
=
\max
\left(
S_0\hat r_t,
\epsilon
\right)
}
$$

三个尺度量的直观含义为：

$$
\boxed{S_0=\text{整段视频的人体基准表观骨架尺度}}
$$

$$
\boxed{\hat r_t=\text{当前帧相对于基准状态的整体表观缩放比例}}
$$

$$
\boxed{S_t=S_0\hat r_t=\text{当前帧人体的表观骨架尺度}}
$$

当人物靠近摄像机时，$\hat r_t>1$ 且 $S_t>S_0$；当人物远离摄像机时，$\hat r_t<1$ 且 $S_t<S_0$。

## 7. 相对坐标原点

当前相对坐标原点的横坐标取双髋中心：

$$
o_t^x
=
\frac{x_{t,11}+x_{t,12}}{2}
$$

纵坐标取有效脚踝中图像位置更靠下的点：

$$
o_t^y
=
\max(y_{t,15},y_{t,16})
$$

因此：

$$
\mathbf o_t
=
\begin{bmatrix}
o_t^x\\
o_t^y
\end{bmatrix}
$$

对原点使用 5 帧中位数平滑：

$$
\hat{\mathbf o}_t
=
\operatorname{median}
\left\{
\mathbf o_u\mid |u-t|\le2
\right\}
$$

当髋或脚踝无效时，分别退化为有效关节点横坐标均值和包围框底部。

## 8. 骨长归一化相对坐标

整个归一化过程可以理解成两个连续步骤。第一步先除以表观缩放比例，将当前帧恢复到该序列的基准大小：

$$
\mathbf p_{t,i}^{canonical}
=
\frac{
\mathbf p_{t,i}^{abs}-\hat{\mathbf o}_t
}{\hat r_t}
$$

第二步再除以序列统一标尺 $S_0$，将基准大小转换成无量纲标准骨架：

$$
\mathbf p_{t,i}^{rel}
=
\frac{
\mathbf p_{t,i}^{canonical}
}{S_0}
$$

将两步合并，并利用 $S_t=S_0\hat r_t$，得到当前代码实际使用的公式：

$$
\boxed{
\mathbf p_{t,i}^{rel}
=
\frac{
\mathbf p_{t,i}^{abs}
-
\hat{\mathbf o}_t
}{S_t}
}
$$

因此，减去原点用于消除人物在画面中的平移；除以 $\hat r_t$ 用于近似撤销当前帧的整体表观透视缩放；继续除以 $S_0$ 用于消除不同人物和不同视频的基准大小差异。最终结果不是恢复真实三维人体尺寸，而是将二维骨架映射到统一的无量纲尺度。

当前代码最后还对相对关节点使用 5 帧时间平滑，并将无效关节点重新置零。

## 9. 骨长归一化 CoM

在相对坐标中计算骨段质心：

$$
\mathbf C_{t,b}^{rel}
=
\mathbf p_{t,i_b}^{rel}
+
\kappa_b
\left(
\mathbf p_{t,j_b}^{rel}
-
\mathbf p_{t,i_b}^{rel}
\right)
$$

人体相对 CoM 为：

$$
\boxed{
\mathbf C_t^{rel}
=
\sum_{b=1}^{B}
w_b\mathbf C_{t,b}^{rel}
}
$$

由于 $\sum_b w_b=1$，该式等价于：

$$
\boxed{
\mathbf C_t^{rel}
=
\frac{
\mathbf C_t^{abs}
-
\hat{\mathbf o}_t
}{S_t}
}
$$

因此不再需要额外使用人体高度 $H_t$ 对 CoM 进行归一化。

## 10. 尺度归一化 CoM 速度

不能直接对 $\mathbf C_t^{rel}$ 求时间差分，因为逐帧变化的原点 $\hat{\mathbf o}_t$ 和尺度 $S_t$ 会被引入速度。

先在绝对坐标中计算 CoM 中心差分：

$$
\mathbf V_t^{abs}
=
\frac{
\mathbf C_{t+1}^{abs}
-
\mathbf C_{t-1}^{abs}
}{2\Delta t}
$$

再使用当前骨长尺度归一化：

$$
\boxed{
\mathbf V_t^{norm}
=
\frac{
\mathbf C_{t+1}^{abs}
-
\mathbf C_{t-1}^{abs}
}{2\Delta t\,S_t}
}
$$

其中 $\Delta t=1/fps$。如果训练和推理均按统一采样帧处理，也可以设 $\Delta t=1$，此时速度单位为“标准骨长/采样帧”。

首尾帧分别使用前向差分和后向差分。

## 11. 原始生物力学 XCoM

线性倒立摆的固有频率为：

$$
\omega_0
=
\sqrt{\frac{g}{l}}
$$

原始 XCoM 方程为：

$$
XCoM_t
=
CoM_t
+
\frac{\dot{CoM}_t}{\omega_0}
$$

等价形式为：

$$
XCoM_t
=
CoM_t
+
\sqrt{\frac{l}{g}}
\dot{CoM}_t
$$

其中 $g$ 为重力加速度，$l$ 为支撑点到真实 CoM 的有效高度。

## 12. 二维表观 XCoM

单目 2D 视频没有真实米制坐标、地面平面和有效质心高度，因此不直接使用真实 $g$ 和 $l$。使用正值可学习速度外推系数：

$$
\lambda
=
\operatorname{softplus}(\theta_\lambda)
+
\epsilon
$$

二维表观 XCoM 定义为：

$$
\boxed{
\mathbf {XCoM}_t^{rel}
=
\mathbf C_t^{rel}
+
\lambda\mathbf V_t^{norm}
}
$$

代入 CoM 和速度后：

$$
\boxed{
\mathbf {XCoM}_t^{rel}
=
\frac{
\mathbf C_t^{abs}-\hat{\mathbf o}_t
}{S_t}
+
\lambda
\frac{
\mathbf C_{t+1}^{abs}-\mathbf C_{t-1}^{abs}
}{2\Delta t\,S_t}
}
$$

第一项描述当前归一化质心位置，第二项描述归一化后的质心运动趋势。

## 13. 与 BoS 和 AMoS 的关系

BoS 边界必须使用相同原点和尺度变换：

$$
x_t^{left,rel}
=
\frac{x_t^{left,abs}-\hat o_t^x}{S_t}
$$

$$
x_t^{right,rel}
=
\frac{x_t^{right,abs}-\hat o_t^x}{S_t}
$$

二维表观稳定裕度为：

$$
AMoS_t
=
\min
\left(
XCoM_t^{x,rel}-x_t^{left,rel},
x_t^{right,rel}-XCoM_t^{x,rel}
\right)
$$

$AMoS_t>0$ 表示 XCoM 位于近似支撑区内，$AMoS_t<0$ 表示 XCoM 已越过近似支撑边界。

## 14. 统一训练方式

所有可学习物理参数记为：

$$
\Theta_{phy}
=
\left\{
\delta_b,
\theta_{\kappa_b},
\theta_\lambda
\right\}
$$

平衡特征进入辅助时序分支，并与 CTR-GCN 特征融合。整个系统只使用当前 60 类交叉熵损失：

$$
\boxed{L=L_{CE}^{60}}
$$

梯度路径为：

$$
L_{CE}
\rightarrow
\mathbf {XCoM}^{rel}
\rightarrow
\mathbf C^{rel},\mathbf V^{norm}
\rightarrow
w_b,\kappa_b,\lambda
$$

物理合理性由正值、归一化和有界参数化保证，不再分别增加质量、平滑或排序损失。

## 15. 符号汇总

| 符号 | 含义 |
|---|---|
| $\mathbf p_{t,i}^{abs}$ | 绝对二维关节点 |
| $\mathbf p_{t,i}^{rel}$ | 骨长归一化相对关节点 |
| $s_{t,i}$ | 关节点置信度 |
| $M_{t,i}$ | 关节点有效掩码 |
| $\mathbf l_{t,b}$ | 第 $b$ 个骨段的向量 |
| $L_{t,b}$ | 当前帧骨段长度 |
| $L_b^{true}$ | 人体真实三维骨长，二维视频中不可直接获得 |
| $L_b^{proj,ref}$ | 第 $b$ 条连杆在该序列中的二维投影参考长度 |
| $z_t$ | 人物远近产生的当前帧整体表观缩放 |
| $z_0$ | 整段视频基准状态下的整体表观缩放 |
| $q_{t,b}$ | 骨段朝向摄像机产生的二维投影缩短比例 |
| $S_0$ | 整段视频的人体基准表观骨架尺度 |
| $r_{t,b}$ | 当前骨长与参考骨长的比例 |
| $\hat r_t$ | 当前帧相对于序列基准状态的整体表观缩放比例 |
| $S_t$ | 当前帧人体的表观骨架尺度，$S_t=S_0\hat r_t$ |
| $\hat{\mathbf o}_t$ | 平滑后的相对坐标原点 |
| $\kappa_b$ | 骨段质心位置比例 |
| $w_b^{prior}$ | 骨段质量先验 |
| $w_b$ | 可学习骨段质量比例 |
| $\mathbf C_{t,b}^{abs}$ | 绝对骨段质心 |
| $\mathbf C_t^{abs}$ | 绝对人体 CoM |
| $\mathbf C_t^{rel}$ | 骨长归一化人体 CoM |
| $\mathbf V_t^{abs}$ | 绝对 CoM 速度 |
| $\mathbf V_t^{norm}$ | 骨长归一化 CoM 速度 |
| $\Delta t$ | 相邻采样帧的时间间隔 |
| $\lambda$ | 可学习速度外推系数 |
| $\mathbf {XCoM}_t^{rel}$ | 二维表观外推质心 |
| $AMoS_t$ | 二维表观动态稳定裕度 |
| $\epsilon$ | 防止除零的小常数 |

## 16. 核心公式

最终核心表达式为：

$$
\boxed{
\mathbf {XCoM}_t^{rel}
=
\underbrace{
\frac{
\mathbf C_t^{abs}-\hat{\mathbf o}_t
}{S_t}
}_{\text{骨长归一化质心位置}}
+
\underbrace{
\lambda
\frac{
\mathbf C_{t+1}^{abs}-\mathbf C_{t-1}^{abs}
}{2\Delta t\,S_t}
}_{\text{骨长归一化质心运动趋势}}
}
$$

由于该结果来自单目二维骨架而非 3D 动作捕捉和力平台测量，论文中应称为 **2D apparent XCoM**，不能直接称为真实生物力学 XCoM。
