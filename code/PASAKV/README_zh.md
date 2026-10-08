# PASA-KV 匿名代码快照

这是论文 *PASA-KV: Attacking Action-Facing Visual Routes in World Action Models* 的精简、模型无关参考实现。它包含 singleton K/V probe、task-balanced sampler、action-facing Tail-AV 评分、确定性 Top-k 选择、fresh joint K/V patch training、patch 投影和评估 trial-grid 检查；不包含作者/机构信息、模型权重、数据集、模拟器资产、实验结果或上游模型代码。

对候选层 l，probe 目标为 J_KV(l)=0.5*(D_K+D_V)，其中 D_Z=||Z_patch-Z_clean||_F/max(||Z_clean||_F,eps)。Tail-AV 在完整 valid key 集上做 softmax，再切出 visual-token 列且不重新归一化；I=A_visual*||V_visual||_2。分数是每个 task 上 patched/clean top 0.5% influence 均值的 log ratio，再对 task 等权平均。排序按 score 降序、layer ID 升序打破并列，不使用 rollout 结果。

最终 patch 从 fresh pixels 和 optimizer state 初始化；目标为所选层 J_KV 的均值。每个 exposure 一次 AdamW update，常数学习率，无梯度裁剪/累积，每步后 clamp 到 [0,1]。

论文配置摘要：FastWAM-Direct 为 30 层、50x50 current-primary patch、坐标 (5,160)；LingBot-VA 为 30 层、29x29 current-agent-view patch、坐标 (3,91)，video endpoint 0.6 后接 10 个 action steps；Cosmos Policy 为 28 层、50x50 current-primary patch、坐标 (5,160)，action slot 4 查询 visual slot 3。实际 token index 必须由对应 adapter 从模型实现获得。

上游框架需单独安装并实现 src/pasakv/adapters/base.py 的 PASAAdapter。Adapter 必须提供完整 key 维度上的 attention-ready Q/K/V、可广播的有效 key mask 与 additive attention bias、实际 visual-token indices，并对 clean/patch forwards 重放相同随机张量；相同全局 seed 本身不足以证明配对。Probe training 与 scoring ID 会检查互斥，final-training 若误用 scoring ID 会立即拒绝该样本。

安装和运行 CPU 测试：

~~~bash
python -m pip install -e ".[test]"
pytest
~~~

参考 Top-k 层集仅供对应 checkpoint 的复核，正式运行须从 frozen Probe100 ranking 选择。论文承诺的 FastWAM checkpoint 将另行发布到 Hugging Face，本仓库不含 .pt。当前未设置许可证；公开前须选定并检查上游许可兼容性。
