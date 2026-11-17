# 测试截图清单

截图由 `scripts/capture_screenshots.py` 与 `scripts/capture_integration.py` 用 Microsoft Edge 无头模式生成，
窗口 1600×1040，均为真实运行页面（非设计稿）。

| 文件 | 页面 | 证明什么 |
| --- | --- | --- |
| `01-chat-product.png` | Demo 问答对比 | 产品说明表格 + 引用 [n] + 检索片段回链 |
| `02-chat-policy.png` | Demo 问答对比 | 制度解释格式（结论 + 分点 + 引用） |
| `03-chat-troubleshoot.png` | Demo 问答对比 | 故障处理格式（现象/原因/步骤） |
| `04-chat-conflict.png` | Demo 问答对比 | 冲突召回：两种口径 + 请用户确认 |
| `05-chat-refusal.png` | Demo 问答对比 | 无召回拒答：缺失信息 + 建议 |
| `06-dataset.png` | Demo 数据集 | 3,547 条、9 类能力、样本浏览 |
| `07-metrics.png` | Demo 评测结果 | 5 组对比、综合分、分能力 |
| `08-integration.png` | Demo 接口与集成 | OpenAI 兼容端点 + 2.9 配置 |
| `09-29-chat.png` | 2.9 知识库平台 | 接入 `kb-finetuned-dph` 后的真实问答与引用 |
| `10-29-config.png` | 2.9 模型配置 | base_url `:8802/v1`、model `kb-finetuned-dph` |
| `11-report-preview.png` | HTML 报告首页 | 报告排版与核心指标 |

## 重新生成

```bash
python run_demo.py                                  # 8802
python scripts/capture_screenshots.py               # 01~08
# 可选：2.9 起在 8803 并配置好微调端点后
python scripts/capture_integration.py               # 09~10
```

> Edge 无头模式在受限沙箱下可能因命名管道权限无法启动；普通桌面环境直接运行即可。
