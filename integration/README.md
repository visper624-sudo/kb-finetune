# 与 2.9 知识库平台集成

## 为什么可以“零改造”

2.9 的问答链路是：

```
检索（向量 Top20 + BM25 Top10 → RRF）→ 四维数据权限鉴权裁剪
→ Prompt 编排（片段带 [n] 编号）→ LLM（OpenAI 兼容 / 本地抽取）
→ SSE 流式 → 引用卡片
```

本项目的 Demo 服务同时实现了 **OpenAI 兼容的 `/v1/chat/completions`**：
它从 2.9 拼好的 user 消息中反解出「问题 + 片段列表」，交给微调模型生成，再按 OpenAI SSE 协议返回。
所以 2.9 只需要把「模型提供方」改成 openai，其余代码与配置全部不动。

## 三步接入

1. 启动本仓库 Demo（默认 8802）：

   ```bash
   python run_demo.py
   ```

2. 启动 2.9（示例用 8803）：

   ```bash
   cd ../知识库管理平台
   python run.py --port 8803
   ```

3. 登录 2.9 管理后台 → 组织与系统配置 → 模型与系统参数：

   | 配置项 | 值 |
   | --- | --- |
   | 推理模式 | OpenAI 兼容接口 |
   | 对话 Base URL | `http://127.0.0.1:8802/v1` |
   | 对话 API Key | `any`（本服务不校验） |
   | 对话模型 | `kb-finetuned-dph` |

   或直接跑本目录的一键验证（会登录并 PUT 配置 + 提问）：

   ```bash
   python integration/check_integration.py --kb http://127.0.0.1:8803
   ```

## 边界与注意事项

- **微调不替代鉴权**：2.9 在拼 Prompt 前已经用 `permissions.split_candidates` 裁掉了无权片段；
  微调模型只能看到已授权的上下文。不要为了“让回答更完整”而绕过鉴权。
- **Prompt 必须一致**：`config/prompts.py` 是训练/推理/2.9 接入三处的唯一 system prompt 来源。
- **上下文编号**：2.9 的 `[1]..[6]` 与本服务解析后的 `n` 一一对应；引用卡片仍由 2.9 渲染。
- **流式**：服务默认逐段输出 SSE；2.9 的 `stream_openai` 可以直接消费。
- **超时/降级**：如果 Demo 服务不可用，2.9 会走它自己的降级逻辑（本地抽取式引擎），不会卡死。

## 验证记录

`../screenshots/09-29-chat.png`：2.9 问答工作台接入 `kb-finetuned-dph` 后的真实回答与引用；
`../screenshots/10-29-config.png`：2.9 的模型配置页（base_url / model 可见）。
