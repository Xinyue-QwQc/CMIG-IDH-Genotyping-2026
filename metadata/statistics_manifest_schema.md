# 统计脚本输入清单与运行说明

脚本 `main_statistics.py` 只读取本地预测并计算统计，不训练、不推理、不连接服务器。正式分析固定为 20,000 次 bootstrap、随机种子 20260920、52 项 Holm 校正；`--smoke` 只用于验证，其输出不得用于论文。

```json
{
  "schema_version": 1,
  "mapping_path": "../../审稿补充材料_旧实验整理_2026-09-20/evidence/data_case_mapping.json",
  "seed": 20260920,
  "bootstrap": 20000,
  "ours_model": "Ours",
  "planned_models": [
    {"model": "Ours", "display_name": "Ours"},
    {"model": "EViT", "display_name": "EViT"}
  ],
  "cohorts": [
    {"id": "Internal", "mapping_dataset": "Internal", "mapping_split": "Test"},
    {"id": "External", "mapping_dataset": "External", "mapping_split": "Test"}
  ],
  "predictions": [
    {
      "id": "ours_internal",
      "model": "Ours",
      "display_name": "Ours",
      "cohort": "Internal",
      "status": "selected",
      "path": "predictions/ours_internal.csv",
      "sha256": "完整64位SHA256"
    },
    {
      "id": "ours_external_missing",
      "model": "Ours",
      "cohort": "External",
      "status": "missing",
      "path": null,
      "reason": "尚未完成导出"
    }
  ]
}
```

上例只展示字段结构。正式 `planned_models` 必须填写 Ours 和全部 13 个基线，两个队列；未提供预测的计划模型仍然占主表与比较表行，显示 NA。`status` 为 `selected` 的文件才进入计算；其他状态与 `reason` 保留在输出中。每个 `model,cohort` 组合最多一条记录，每条预测的 `id` 唯一。其他来源证据字段可保存在条目中，会原样保存在 `manifest_used.json`。

路径可为绝对路径，也可相对于 manifest 所在目录。`mapping_path` 使用旧材料 `data_case_mapping.json` 的 `cases` 字典结构。脚本按队列对应 dataset 和 Test split 形成完整预期 ID 集；患者组使用映射中的 `patient_group`，无此字段时使用 `patient_id`，不以未知记录 ID 替代患者。映射中不存在的患者、组内标签冲突、预测缺失/多余 ID、重复 ID、标签冲突、非有限分数、超出 [0,1] 的分数、哈希不一致均拒绝进入计算，输出具体原因。

CSV 默认字段为 `ID,idh_truth,pred,pred_class`。正类为 IDH-mutant=1。保存类别必须等于 `pred > 0.5`，精确 0.5 归 WT。可为单条预测配置 `score_column: "score"`，或配置 `columns: {"id":"ID", "label":"idh_truth", "score":"score", "class":"pred_class"}`。若存在 `patient_group` 和 `source` 列，也须与映射一致。

PowerShell 运行：

```powershell
python .\tools\test_main_statistics.py
python .\tools\main_statistics.py --manifest .\inputs\selected_predictions.json --out .\results
python .\tools\build_delivery.py
```

正式命令必须在本补充文件夹下运行。`--bootstrap 500 --smoke` 可以检查程序与输入，其输出请指向单独的 smoke 文件夹。缺失预测是显式部分结果，退出码为 0；已提供但校验错误的文件还会令退出码为 2；manifest 本身结构错误立即报错。

输出包括：

- `metrics_long.csv`：每个模型/队列的 9 个指标、逐项 95% CI、有效重采样数和警告。
- `main_results.csv`、`main_table_Internal.md`、`main_table_External.md`：宽表及便于核对的主表。
- `paired_comparisons.csv`：Ours 减去每个基线的 AUC、ACC 差值、95% CI、原始和 Holm 校正 p 值；差值同时保存比例与百分点。
- `input_validation.csv`：输入校验、SHA256、分类规则、病例与患者数、混淆矩阵；未纳入条目保留原因。
- `patient_index_*.csv` 与 `bootstrap_patient_weights_*.npz`：按 ID 排序的病例到患者映射、每次抽样的患者次数。每个患者的全部扫描始终使用相同次数。
- `bootstrap_metrics.npz`：每个已纳入模型的全部 bootstrap 指标；数组对应关系存于 `analysis_summary.json`。
- `analysis_summary.json`、`manifest_used.json`：协议、版本、输入及程序哈希、有效分析数量和解释限制。

分类指标无定义时输出 NA 并记录有效次数。完全相同预测的差值为 0、p=1；非零而 bootstrap 差值完全恒定的退化比较保留差值/区间但 p 为 NA，不把最小 Monte Carlo 值误作可靠证据。Holm 始终保留 52 项家族，尚不可检验项在内部以 p=1 占位，展示仍为 NA。

`build_delivery.py` 默认读取正式结果及选定清单，生成 `reports/统计结果报告.md`、`paper_revision/main_tables.tex`、`paper_revision/statistics_methods.tex` 和 `paper_revision/审稿回复草稿.md`，不会覆盖原始论文。它拒绝 smoke 结果、清单改变后未重算的结果以及指标/比较行数不完整的输入。正式输入 28/28 可用才会在 `reports/delivery_summary.json` 中标为 `complete`，否则明确为 `partial` 并列出缺项。它支持 `--manifest`、`--results`、`--out-root` 指定独立位置。
