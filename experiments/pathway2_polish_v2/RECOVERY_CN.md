# seed44 入口精度复核修复

该分支以 `3ea683c2ab115b6937d0d1b2a5834bebfd015b0d` 为父提交。
旧 polish 分支与 stable 分支保持原状。优化器、目标函数和预算保持原状。

旧 stationarity 审计沿用 PyTorch 2.8 默认的 cuDNN TF32 允许设置，新 polish
`runtime.load_context()` 显式关闭 cuDNN TF32。原代码将不同精度设置下的 CE
直接比较，seed44 在优化开始前触发阈值。旧报错没有打印实际 CE，因而尚不能
仅凭这条报错断言精度差异就是全部原因。

新增 `entry_audit.py` 在优化器创建前完成以下操作。

1. 显式设置 matmul TF32=false、cuDNN TF32=true，复现旧 CE 和旧 g/b 向量。
   CE 的原阈值 `1e-8 + 5e-5 * abs(reference_ce)` 保持不变。
2. 两处 TF32 均关闭，重复完整快照，要求 CE 与梯度向量完全一致。
3. FP64 副本复核，要求严格 FP32 与 FP64 CE 一致到同一形式的阈值，
   每个 g/b 向量的差异不超过 FP64 向量范数的 10%。
4. 无论检查通过或失败，记录 `entry_audit.json`。失败禁止开始优化。
   `precision_mismatch_explains_old_failure=true` 只在所有检查通过且
   严格 FP32 CE 仍触发旧跨精度门槛时写入。

`prepare --retry-from ORIGINAL_OUT` 检查原 v2 提交、输入与源码哈希、旧 smoke、
旧 seed42 完成文件，并拒绝覆盖已有 seed44 完成结果。只生成 seed44 一个 cell，
新的 smoke 也针对 seed44。所有输出写入新目录。已完成 seed42 不重新运行，
不复制它的完成标记到新实验。两个 seed 应分别引用其原始输出和提交。

旧梯度向量 `gradient_vectors.npz` 留在 Delta，必须仍存在且符合旧完成标记的哈希。
本地下载的小结果包不包含这些大向量，所以不能用下载包代替 Delta 原输出目录。

本修复仍为 FP32 优化、FP64 端点审计。FP64 检查不是严格误差界，也不证明 IF 有效。
不因一次 100 步 polish 的 budget stop 声称已收敛。

数值设置参考：https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html
