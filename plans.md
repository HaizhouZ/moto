# Restoration 集成计划（再修订）

## Summary
在现有 `normal iteration` 骨架上引入显式 `restoration phase`，但求解对象必须明确为 **restoration Lagrangian 导出的 reduced KKT system**，不是“只优化 primal violation”或“只优化 prox objective”。实现上复用当前 `update_approximation -> solve_direction -> correction/IR -> line search -> accept` 主流程，同时把 restoration 的局部 elastic/IPM 状态、残差定义、退出映射做成独立相位语义。

这版按你的两点再收紧：
- `restoration` 的方向、残差、IR、acceptor 全都以 **resto Lagrangian** 为准；
- elastic block 中所有正性变量的 `backup/restore/affine step/fraction-to-boundary/update_ls_bounds` 尽量复用 `ipm_constr` 的已有逻辑，而不是再写一套同构实现。

对齐基线以 [normal_iteration.md](/home/harper/Documents/moto/normal_iteration.md) 为准，必须始终区分三层对象：

- 原问题
- local reduced soft/ineq 子问题
- 最终 Riccati reduced 子问题

restoration 的实现和验证也必须沿这三层来描述，不能把它们混成“一个 restoration 问题”。

## Key Changes
### 0. 相位隔离下的复用约束
- “复用 normal 代码”不是目标本身；只有 **phase-safe 的复用** 才允许进入实现。
- 任何被 restoration 复用的 normal 逻辑都必须满足：
  - 输入/输出语义在 `normal` 与 `restoration` 下完全一致，或
  - 被显式包在 phase gate 内，保证另一相位的状态不会被隐式推进
- 如果某段 normal 逻辑会修改 outer NLP 专属状态，例如：
  - outer IPM slack / multiplier
  - outer filter history
  - outer barrier parameter `settings.ipm.mu`
  - outer KKT / objective 语义
  那么 restoration 只能复用其“无副作用的数学内核”，不能直接复用整段控制流。
- 因此 restoration 接入时必须优先检查：
  - `backup/restore`
  - `apply_affine_step`
  - `update_ls_bounds`
  - predictor/corrector
  - trial acceptance / filter update
  是否会同时推进 normal 与 restoration 两套状态；若会，则必须拆分或显式短路。
- 代码层面优先把这类 phase gate 收敛成少量公共 helper，而不是把 `if (settings.in_restoration)` 散落到各处，避免未来继续复用时再引入混淆。
- restoration 的 entry/exit cleanup 也必须集中到 helper：outer dual 恢复、bound/slack 映回、equality multiplier reset 不能散落在主循环和 line-search 分支里分别实现。
- 临时按另一相位解释同一批数据时，也必须走统一 helper；不能在业务代码里手动翻转 `settings.in_restoration` 再自行恢复。

### 1. 求解目标固定为 restoration Lagrangian
- restoration 子问题固定采用 [restoration.md](/home/harper/Documents/moto/restoration.md) 中的 stagewise restoration Lagrangian：
  - `obj_R(w)` 作为 restoration base objective
  - `rho_eq * (p_c+n_c) + rho_ineq * (p_d+n_d)` 作为 exact L1 elastic penalty
  - `-mu_bar * log(...)` 作为 restoration barrier
  - `lambda_f, lambda_c, lambda_d` 进入 restoration Lagrangian 约束项
- `solve_direction()` 在 restoration phase 中求的是该 Lagrangian 线性化/二次化后的 reduced KKT，不是单纯最小化 `theta_R`，也不是 outer NLP 的 barrier objective。
- prox 项仍作为 `cost` 存在，但它属于 `obj_R(w)`，因此必须进入：
  - restoration `cost_`
  - restoration `cost_jac_`
  - restoration base `lag_ / lag_jac_`
  - restoration prox Hessian storage
- outer 原始用户 cost 在 restoration 中只保留为“退出时的 outer 可接受性判断输入”，不进入 restoration direction。

这里的层次固定为：

- restoration 原问题：
  restoration Lagrangian with exact L1 elastic penalty, local elastic variables, and barrier
- restoration local reduced 子问题：
  eq elastic / ineq elastic block 的局部 condensation
- restoration Riccati 子问题：
  将这些 local reduced 贡献写入 stage QP 后，再进入 nullspace / Riccati reduction

### 2. 组装分层与 corr 生命周期
- normal 与 restoration 共用 raw evaluation；区别只在 active QP assembly。
- restoration assembly 固定分三层：
  - base hard constraints：仅 `__dyn`
  - base restoration cost：`obj_R(w)`，当前先是 prox
  - condensed elastic/IPM corrections：来自 eq elastic 与 restoration-owned ineq 的 Schur complement
- 所有 condensed 一阶项统一写入 `lag_jac_corr_`。
- 所有 condensed 二阶项统一写入 `hessian_modification_`。
- prox 二阶项属于 restoration base cost Hessian，不能被算成 corr；当前通过独立的 prox diagonal storage 接入 Riccati，而不混入 local reduced correction Hessian。
- `ns_factorization` 只消费组装结果；不再在 factorization 阶段临时生成 restoration 数学，避免 base/corr 生命周期混淆。

对齐 normal 的要求是：

- local reduced 子问题的二阶项进入 solver-owned correction Hessian 通道
- base objective 的二阶项保持独立语义
- Riccati 子问题只消费组装后的 stage QP，不重新推导 local reduced 数学

### 3. 乘子与相位语义
- `dense().dual_[cf]` 继续复用为“当前相位该约束的 multiplier 真值”：
  - normal phase：outer NLP multipliers
  - restoration phase：restoration multipliers
- 这不是双写模型。避免混淆的机制是：
  - restoration 入口时先快照 outer dual 到 `restoration_state::outer_dual_backup`
  - restoration 运行时，`dual_` 全部按 restoration 语义解释
  - restoration 退出时统一做 cleanup / reset / rebuild，再恢复 outer NLP 语义
- `__dyn` dual 始终保留 hard dynamics 含义，不 elastic 化。
- `restoration_state` 只存 restoration 局部变量、备份和工作缓存，不再额外复制一份全量 constraint multipliers。

### 4. 残差、IR 与 objective 绑定
- phase-aware residual assembly 必须显式绑定当前 objective。
- normal phase 残差保持现状。
- restoration phase 残差定义固定为：
  - `w`-stationarity：来自 **restoration Riccati 子问题** 对应的 reduced stationarity
  - local stationarity：`p/n/t/...` 对应的 stationarity residual
  - complementarity：restoration barrier 正性块 residual
  - primal residual：`F(w)`, `c-p+n`, `g+t-p_d+n_d`
- `compute_kkt_residual()` 不能把 normal cost 或 normal barrier objective 的导数带进 restoration residual。
- iterative refinement 控制流继续复用，但 residual checker 和 correction RHS 必须按 phase 切换。
- restoration 下的 IR stop check 必须显式纳入 local elastic block 的 stationarity / complementarity，避免只看 `w`-stationarity 就误判“已收敛”。
- 在 reduced RHS 还没按 restoration 局部块完整推导前，不能把 normal-phase IR correction 直接施加到 restoration 上；宁可先禁用 restoration IR，也不能用错误 RHS 把 Riccati 递推炸掉。
- restoration line search 的 inner acceptor 使用 restoration objective 与 restoration primal violation；outer filter 只在 restoration 成功返回判断时介入。

这里的对齐原则固定为：

- IR 真正减小的是 restoration Riccati 子问题的 reduced correction residual
- `compute_kkt_info()` / `print_stats()` 是 restoration phase 的公开摘要，不要求与 IR 内部向量逐分量相同
- 但两者必须来自同一 restoration phase，而不能一个看原问题、一个看 Riccati 子问题

### 5. 复用 ipm_constr 的正性/步长逻辑
- 不为 elastic slack/dual 再写一套独立的正性边界逻辑。
- 从 `ipm_constr` 中提取一个共享的“正性块更新 helper”，统一服务：
  - `ipm_constr`
  - eq elastic block 里的 `p_c, n_c, nu_p_c, nu_n_c`
  - restoration-owned ineq block 里的 `t, p_d, n_d, nu_t, nu_p_d, nu_n_d`
- 该 helper 固定复用现有 `ipm_constr` 的行为模式：
  - `backup_trial_state`
  - `restore_trial_state`
  - `apply_affine_step`
  - `update_ls_bounds`
  - fraction-to-boundary 风格的 `alpha_max` 计算
- 共享 helper 只处理“正性变量和其对偶变量的步长/边界/backup-restore”，不碰每类约束的局部 KKT 数学。
- 这样 elastic block 与 normal IPM 在步长边界语义上保持一致，减少维护分叉。

### 6. 入口与退出
- 入口条件保持：outer globalization 发生 `tiny_step` failure 且 outer primal infeasibility 仍大于容差。
- outer `tiny_step` 的判定阈值必须和 restoration trigger 共用同一套动态 `alpha_min` 逻辑；不能一边在 `update()` 里想触发 restoration，另一边 line search 永远不给出 `tiny_step`。
- 入口动作固定为：
  - 快照 outer dual state
  - 初始化 `mu_bar`
  - 初始化 eq elastic 与 restoration-owned ineq block
  - 切换 `dual_` 到 restoration 语义
  - 建立 prox reference，并把 prox 作为 restoration cost
- 成功退出 restoration 时：
  - primal iterate 保留
  - bound multipliers/slacks 按设计映回 normal storage
  - equality multipliers 依 `constr_mult_reset_threshold` 规则清零或最小二乘重建
  - cleanup 后重新按 outer NLP 语义计算 KKT
- 失败退出时：
  - 一样执行 cleanup，禁止 restoration multiplier 语义残留
  - 再按 outer NLP 语义计算最终 `kkt_last`

## Interfaces / Types
- 公开新增或恢复：
  - `ns_sqp::settings_t::restoration`
  - `ns_sqp::iter_result_t::restoration_failed`
- 内部新增：
  - `iteration_phase`
  - `restoration_state`
  - `restoration_eval_info`
  - phase-aware QP assembler
  - phase-aware residual assembler
  - 共享 positivity-step helper
- `restoration_state` 最小字段固定包含：
  - `mu_bar`
  - `outer_dual_backup`
  - eq elastic local state
  - restoration-owned ineq local state
  - prox refs / scales
  - local trial backups
  - restoration metrics cache

## Test Plan
### 1. 数学正确性单元测试
- eq elastic local KKT：
  - condensation 前后线性系统等价
  - recovery 的 `delta lambda/delta p/delta n/delta nu` 满足原线性化方程
  - 覆盖 `lambda_reg = 0` 与 `lambda_reg > 0`
- restoration-owned ineq local KKT：
  - condensation 与 recovery 自洽
  - `t/p_d/n_d` 局部系统对原线性化方程闭合
- restoration Lagrangian 一致性：
  - 验证 `w`-stationarity 使用的是 restoration objective + restoration multipliers
  - 验证 outer 原始用户 cost 不进入 restoration direction
  - 验证 prox 作为 restoration cost 的 value/grad/hess 全部落在正确通道
- corr 生命周期：
  - condensed 一阶项只进 `lag_jac_corr_`
  - condensed 二阶项只进 `hessian_modification_`
  - prox Hessian 属于 base cost Hessian，不计入 corr

### 2. 逻辑复用与不混淆测试
- 正性块 helper 复用测试：
  - `ipm_constr` 与 restoration elastic block 对同类负方向步长给出一致的 `alpha_max`
  - `backup/restore/apply_affine_step` 行为一致
- 相位语义测试：
  - restoration 期间 `dual_` 表示 restoration multipliers
  - 退出后 `dual_` 全部恢复 outer NLP 语义
  - 任一退出路径后都不能残留 restoration multiplier 语义

### 3. 退出乘子测试
- 成功退出后验证：
  - bound multipliers 按 `bound_mult_reset_threshold` 规则重置
  - equality multipliers 按 `constr_mult_reset_threshold` 规则清零或最小二乘重建
  - cleanup 后 outer KKT 重新计算且不依赖 restoration 本地状态
- 失败退出后验证：
  - restoration multiplier 不残留
  - outer dual 按设计恢复/重建
  - `kkt_last` 使用 outer NLP 语义重算

### 4. 集成与回归
- restoration 未启用或未触发时，normal 路径结果与当前主干一致。
- 构造 tiny-step failure 样例，验证：
  - 能进入 restoration
  - 方向来自 restoration Lagrangian
  - inner acceptor/IR/residual 走 restoration 语义
  - success / restoration_failed / infeasible_stationary 三类出口正确
- 普通样例回归，确认 normal 功能无退化。

## Assumptions / Defaults
- 第一版按全量 restoration 实现，不做 eq-only 缩减版。
- restoration 优化对象固定是 restoration Lagrangian 的 reduced KKT，不允许退化成只降 `theta_R` 的启发式过程。
- prox 作为 restoration `cost`，但不污染 outer 原始用户 cost。
- `dual_` 复用为当前相位 multiplier 真值；避免混淆依靠严格的入口快照和退出 cleanup，而不是双份存储。
- elastic 正性变量的步长边界、trial backup/restore、affine update 尽量复用 `ipm_constr` 逻辑；若必须抽象公共 helper，则 `ipm_constr` 与 restoration 都改为调用同一 helper。
