# 失败案例分析（自动抽取）

## test-topic-00008（product，通过 1/5）

**问题**：请用表格列出出差住宿费限额？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】请用表格列出出差住宿费限额
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 一线城市（北京、上海、广州、深圳）住宿标准 | 每人每晚 500 元。 | [1] |
| 要点 2 | 总监级及以上岗位可在上述标准基础上上浮 50%。 | [1] |
| 要点 3 | 需附审批记录。 | [1] |
| 超标部分 | 个人承担。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00010（product，通过 1/5）

**问题**：出差住宿费限额的档位和标准有哪些？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】出差住宿费限额的档位和标准有哪些
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 一线城市（北京、上海、广州、深圳）住宿标准 | 每人每晚 500 元。 | [1] |
| 要点 2 | 总监级及以上岗位可在上述标准基础上上浮 50%。 | [1] |
| 要点 3 | 需附审批记录。 | [1] |
| 超标部分 | 个人承担。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00012（product，通过 1/5）

**问题**：请列出出差住宿费限额的具体内容？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】请列出出差住宿费限额的具体内容
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 一线城市（北京、上海、广州、深圳）住宿标准 | 每人每晚 500 元。 | [1] |
| 要点 2 | 总监级及以上岗位可在上述标准基础上上浮 50%。 | [1] |
| 要点 3 | 需附审批记录。 | [1] |
| 超标部分 | 个人承担。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00014（product，通过 1/5）

**问题**：出差住宿费限额分别是什么标准？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】出差住宿费限额分别是什么标准
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 一线城市（北京、上海、广州、深圳）住宿标准 | 每人每晚 500 元。 | [1] |
| 要点 2 | 总监级及以上岗位可在上述标准基础上上浮 50%。 | [1] |
| 要点 3 | 需附审批记录。 | [1] |
| 超标部分 | 个人承担。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00016（policy，通过 1/5）

**问题**：出差打车费用有限额吗？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】出差打车费用有限额
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
【结论】网约车单次不超过 100 元，一个月累计不超过800 元 [1]。

【依据】
1. 城际出行：飞行时间 2 小时以内乘坐高铁二等座，2 小时以上可乘坐飞机经济舱 [1]。
2. 总监级及以上可乘坐高铁一等座 [1]。
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少【依据】/【要点】", "缺少分点依据", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00036（policy，通过 1/5）

**问题**：出差餐补标准这块有什么要求？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】出差餐补标准这块有什么要求
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
【结论】出差期间按自然日发放餐补，一线城市 120 元/天，其他城市 100 元/天 [1]。

【依据】
1. 当天往返且出差时长不足 6 小时的不发放餐补 [1]。
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少【依据】/【要点】", "缺少分点依据", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00100（product，通过 1/5）

**问题**：报销审批权限怎么划分？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】报销审批权限怎么划分
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 单笔 5000 元以下 | 直属上级审批。 | [1] |
| 单笔 2 万元至 10 万元 | 增加财务总监审批。 | [1] |
| 单笔 10 万元以上 | 增加总经理审批。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00102（product，通过 1/5）

**问题**：请用表格列出报销审批权限？

**模型回答**：

```
【结论】现有知识库资料无法回答该问题。
【缺失信息】请用表格列出报销审批权限
【建议】请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 单笔 5000 元以下 | 直属上级审批。 | [1] |
| 单笔 2 万元至 10 万元 | 增加财务总监审批。 | [1] |
| 单笔 10 万元以上 | 增加总经理审批。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00103（product，通过 1/5）

**问题**：报销审批权限的档位和标准有哪些？

**模型回答**：

```
【结论】检索到的资料与问题关联度较低，无法给出可靠答案。
【缺失信息】报销审批权限的档位和标准有哪些；检索到的片段只覆盖相邻主题，缺少该问题的关键口径
【建议】请补充《财务制度》类文档，或联系知识管理员确认后再提问。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 单笔 5000 元以下 | 直属上级审批。 | [1] |
| 单笔 2 万元至 10 万元 | 增加财务总监审批。 | [1] |
| 单笔 10 万元以上 | 增加总经理审批。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}

## test-topic-00104（product，通过 1/5）

**问题**：报销审批权限的档位和标准有哪些？

**模型回答**：

```
【结论】检索到的资料与问题关联度较低，无法给出可靠答案。
【缺失信息】报销审批权限的档位和标准有哪些；检索到的片段只覆盖相邻主题，缺少该问题的关键口径
【建议】请补充《财务制度》类文档，或联系知识管理员确认后再提问。
```

**Gold**：

```
| 项目 | 说明 | 依据 |
| --- | --- | --- |
| 单笔 5000 元以下 | 直属上级审批。 | [1] |
| 单笔 2 万元至 10 万元 | 增加财务总监审批。 | [1] |
| 单笔 10 万元以上 | 增加总经理审批。 | [1] |
```

**判定**：{"faithfulness": 0.0, "hallucination": false, "citation_accuracy": 0.0, "citation_coverage": 0.0, "citation_gold": null, "citation_count": 0, "key_point_coverage": 0.0, "format_ok": false, "format_problems": ["缺少表头“项目”", "表格数据行少于 2 行", "缺少引用编号"], "term_ok": null, "refusal_ok": null, "false_refusal": true, "detected_refusal": true, "n_factual": 3, "n_units": 3}
