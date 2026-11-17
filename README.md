# 知识库项目模型微调（项目实战 4.1）

> 从 **2.9 知识库管理平台**的 30 篇制度文档反向合成训练数据，完成「数据构造 → 训练 → 评测 → 集成」全链路：
> 训练模型严格依据检索片段作答，**先引用后结论**，按 4 类格式输出，资料不足时明确拒答。

- **交付形态**：可用训练/评测代码 + 可运行 Demo 控制台 + OpenAI 兼容服务 + 数据与报告 + 测试截图
- **数据来源**：`../知识库管理平台/seed/docs`（30 篇虚构制度文档，PDF×4 / DOCX×4 / MD×12 / TXT×10）
- **运行环境**：Python 3.10+；推理/训练仅需 `numpy`，Demo 与 2.9 集成零额外依赖；LoRA 需 GPU + `transformers/peft`
- **开箱即跑**：`python run_demo.py` → <http://127.0.0.1:8802>

> **仓库说明**：本仓库为演示精简版，仅保留运行 Demo 所需的代码与数据（语料、规格、策略模型、评测摘要、截图、测试集样本）。
> 完整训练/验证数据、逐条预测、LoRA 适配器以及教程/报告类文档未纳入版本库，需要完整复现请使用本地交付包。

---

## 〇、先看结论（TL;DR）

| 项目 | 结果 |
| --- | --- |
| 数据规模 | **3,547** 条（train 2,442 / valid 458 / test 647），拒答类 16.9% |
| 覆盖能力 | 4.1.1 术语/格式、4.1.2 忠实、4.1.3 多片段、4.1.4 引用、4.1.5 拒答 |
| 测试集 | 647 条，按**文档切分**（同一文档不跨 train/test），跨 split 问法重合 0 |
| 综合分 | Baseline **0.447** → DPH **0.814** / **LoRA 0.836** |
| 引用 | DPH 准确率 0.953 / 覆盖率 0.919；LoRA 0.900 / 0.937（Baseline 均为 0） |
| 拒答 | DPH 0.716 / 误拒 0.064；LoRA **0.890 / 0.057**；去掉拒答数据训练后降到 0.126 |
| 格式 | DPH 0.691；LoRA **0.850**，4 类格式可稳定产出 |
| 集成 | 2.9 只改 4 个配置项即可接入；检索、四维鉴权、SSE、引用卡片保持原样 |

> **两条训练路径（都已实测）**：
> ① **本机 DPH**：本机离线，用同一份 SFT 数据训练轻量策略模型（Gate / Format / Evidence Selector），
>    可在 CPU 上复现全部数据与评测；
> ② **云端 LoRA**：Qwen2.5-1.5B-Instruct + bf16 LoRA（r=16, 2 epoch, 612 步, valid_loss 0.218），
>    在 AutoDL RTX 4090 24GB 上完成训练与 647 条测试集评测，产物在 `adapter/lora/`、`eval/results/lora/`。
> 两条路径共用同一 Prompt、同一测试集、同一评测脚本；上表 LoRA 数字为真实云端训练结果。

---

## 一、快速开始

```bash
cd 知识库模型微调
pip install -r requirements.txt          # 只有 numpy；仅跑 Demo/评测也可以直接运行
python run_demo.py                        # http://127.0.0.1:8802
```

Demo 首页提供 4 个页签：**问答对比 / 数据集 / 评测结果 / 接口与集成**。
左侧是「检索后直接拼句子」的基座式回答，右侧是微调模型；可直接看到引用编号、格式、拒答与检索片段。

### 复现训练与评测

```bash
python data/build_dataset.py              # 从 data/corpus/chunks.jsonl 合成 3,547 条样本
python train/distill_policy.py            # 训练 DPH（本机约 1 分钟）
python eval/run_eval.py                   # 6 组对比（含 LoRA，如有 adapter/lora），生成 eval/results/*
python eval/report.py                     # 生成 report/微调报告.html
```

### 云服务器版本（可选，真正训练 LoRA）

云 GPU 的完整步骤（AutoDL / 阿里云 / 腾讯云、上传、下载模型、训练、评测、打包、部署）见
[`cloud/README.md`](cloud/README.md)。最简流程：

```bash
# 上传项目后，在云 GPU 实例上
cd /root/autodl-tmp/知识库模型微调
bash cloud/setup.sh
bash cloud/train_lora.sh          # 下载模型 → 1.5B QLoRA → 评测 baseline/dpf/lora → 报告
bash cloud/package_artifacts.sh   # 打包 adapter/结果，scp 下载
```

本地仍可用 `python run_demo.py` 演示 DPH，不需要云服务器。

### 训练 LoRA（需 GPU / 模型权重）

```bash
pip install -r train/requirements-gpu.txt
python train/train_lora.py --dry-run      # 先校验数据格式与长度
python train/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --output adapter/lora --epochs 2 --cutoff-len 2048 --qlora
python eval/run_eval.py --configs lora --adapter adapter/lora
```

---

## 二、目录结构

```
知识库模型微调/
├── run_demo.py                     # 启动 Demo（8802）
├── run_all.py                      # 一键：数据→训练→评测→报告
├── 一键运行.bat                    # Windows 双击入口
├── requirements.txt                # numpy（推理/训练）
│
├── data/
│   ├── corpus/                     # 从 2.9 抽取的文档/切片（含 source/ 原始 30 篇）
│   ├── specs/                      # 140 个主题、60 无召回、45 弱相关、14 冲突、20 部分可答
│   ├── build_dataset.py            # 数据合成 + 质检 + 切分
│   ├── out/                        # train/valid/test.jsonl + stats/failures
│   └── reports/                    # 质量报告、泄漏检查
│
├── kb/
│   ├── corpus.py                   # 文档/切片/句子三级视图
│   ├── retrieval.py                # BM25 + IDF 稀疏向量 + RRF；策略特征
│   ├── compose.py                  # 4 类格式组装 + 引用 + 拒答/冲突/部分可答
│   ├── format_rules.py             # 格式与数字溯源校验
│   ├── textutil.py                 # 中文分句/分词/数字事实/问题拆分
│   └── terms.py + data/specs/terms.json
│
├── train/
│   ├── train_lora.py               # Qwen LoRA SFT（GPU 路径，支持 QLoRA）
│   ├── distill_policy.py           # DPH 训练（本机实测路径）
│   ├── policy.py                   # Gate/Format/Selector 推理
│   └── config_lora.json            # LoRA 超参（对齐难度方案建议值）
│
├── eval/
│   ├── metrics.py                  # 8 个指标 + 综合分
│   ├── generate.py                 # Baseline / DPH / 3 个消融 / LoRA
│   ├── run_eval.py                 # 评测入口
│   ├── report.py                   # HTML 报告（内联 SVG）
│   ├── human_eval.py               # 120 条人工精评表 + Cohen's κ
│   └── results/                    # predictions / metrics / comparison / cases
│
├── adapter/
│   ├── policy/                     # 已训练的 DPH（model.npz + meta.json）
│   ├── lora/                       # 云端 LoRA adapter（约 86MB，含 tokenizer）
│   └── README.md                   # DPH / LoRA 产物说明
│
├── demo/                           # 标准库 HTTP 服务 + 原生前端（无 CDN）
│   ├── server.py                   # /api/* + /v1/chat/completions
│   └── static/                     # 问答对比 / 数据集 / 评测 / 集成
│
├── cloud/                           # 云 GPU 训练/部署脚本 + 步骤说明
├── integration/                    # 2.9 接入说明 + 一键配置/验证脚本
├── tests/                          # 32 个单元/接口测试
├── screenshots/                    # 11 张真实运行截图（Demo + 2.9 集成 + 报告）
├── report/                         # 微调报告.html（自包含）
└── docs/                           # 报告/数据/评测/验收/术语/答辩/全流程
```

---

## 三、4.1 五项能力怎么落地

| 条款 | 能力 | 数据设计 | 训练/策略 | 指标 |
| --- | --- | --- | --- | --- |
| 4.1.1 | 行业术语与格式适配 | 140 个主题 × 4 类格式；术语表 + 口语别名改写；按格式设计问法 | Format 四分类 + 强线索兜底；答案按格式组装 | 术语 0.882 / 0.824；格式 0.691 / 0.850 |
| 4.1.2 | 检索内容忠实回答 | gold 答案只摘录证据句；逐句数字溯源；片段外事实为 0 | 证据选择器只选上下文中的句子 | 忠实 0.932 / 0.889；幻觉 0.000 / 0.006 |
| 4.1.3 | 多片段信息整合 | 26 组跨文档组合（2~4 个主题），含条件/主次 | integration 格式，多片段证据合并 | 要点覆盖 0.608 / 0.600 |
| 4.1.4 | 证据引用与出处 | 每个结论句/表格行带 `[n]`；编号与上下文一一对应 | 组装器自动写引用；引用可回链片段 | 引用准确 0.953 / 0.900；覆盖 0.919 / 0.937 |
| 4.1.5 | 资料不足拒答 | 60 无召回 + 45 弱相关 + 14 冲突 + 20 部分可答 | Gate 二分类 + 冲突/部分可答规则 | 拒答 0.716 / 0.890；误拒 0.064 / 0.057 |

---

## 四、数据构造（本项目 60% 的工作量）

```
2.9 文档（30 篇）→ 2.9 解析器/切片器（57 个切片，保留 page/para/char）
  → 140 个人工主题（问法 + 证据锚点 + 目标格式 + 标准术语）
  → 自动扩展同义问法/别名问法
  → 检索上下文（top-k，保证 gold 片段在上下文中）
  → 组装 gold 答案（只摘录证据句，句末 [n]）
  → 质检（格式/数字溯源/去重/切分泄漏）
  → train / valid / test（按文档切分）
```

- **正样本**：主题问答 2,845 条、多片段 100 条；覆盖 4 类格式。
- **负样本**：无召回 360、弱相关 158、冲突 42、部分可答 40（含训练/验证/测试）。
- **冲突样本**：构造“旧版/已废止”片段 + 真实制度冲突（差旅报销 15 个工作日 vs 费用报销 30 个自然日）。
- **质量门禁**：答案中数字必须出现在被引用片段；格式机器校验；拒答样本无引用；同一文档不跨 split。
- 详细说明见 [`docs/数据构造说明.md`](docs/数据构造说明.md)，统计见 `data/reports/quality_report.md`。

---

## 五、训练

### 5.1 DPH（本机实测）

| 模型 | 作用 | 验证集表现 |
| --- | --- | --- |
| Gate（逻辑回归） | 判断“资料是否足以回答” | 准确率 0.915、拒答召回 0.931、AUC 0.952、阈值 0.33 |
| Format（softmax） | 4 类格式路由 | 准确率 0.713（steps 0.917 / product 0.737 / policy 0.725 / troubleshoot 0.588） |
| Evidence Selector | 从上下文挑支撑句 | 证据召回 0.705 / 精确率 0.360 |
| Composer | 按格式组装 + 引用 | 引用准确率 0.953 |

训练脚本 `train/distill_policy.py`，产物 `adapter/policy/{model.npz,meta.json}`，日志 `train/train_log.json`。

### 5.2 LoRA（云端实测）

| 项 | 值 |
| --- | --- |
| 基座 | Qwen2.5-1.5B-Instruct |
| 环境 | AutoDL RTX 4090 24GB；torch 2.5.1+cu124 / transformers 4.46.3 / peft 0.13.2 |
| 方式 | bf16 LoRA（QLORA=0），r=16, α=32, dropout=0.05 |
| 超参 | lr=1e-4, 2 epoch, cutoff=2048, batch=1, grad_accum=8 |
| 规模 | train 2,442 / valid 458；612 步；valid_loss 0.218 |
| 产物 | `adapter/lora/`（约 86MB）、`eval/results/lora/`、`train/lora_train_log.json` |
| 结果 | 综合分 **0.836**（DPH 0.814）；格式 0.850、拒答 0.890、引用覆盖 0.937 |

脚本：`train/train_lora.py`、`cloud/train_lora.sh`；步骤见 [`cloud/README.md`](cloud/README.md)。
注意：**训练与推理的 system prompt 必须完全一致**（`config/prompts.py` 单点定义）。

---

## 六、评测结果（test 647 条）

| 指标 | Baseline | DPH | **LoRA** | 无拒答训练 | 无引用 | 无检索 |
| --- | --- | --- | --- | --- | --- | --- |
| 忠实性 ↑ | 1.000 | 0.932 | 0.889 | 0.977 | 0.932 | 0.053 |
| 幻觉率 ↓ | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.680 |
| 引用准确率 ↑ | 0.000 | 0.953 | 0.900 | 1.000 | 0.018 | 0.000 |
| 引用覆盖率 ↑ | 0.000 | 0.919 | 0.937 | 0.966 | 0.013 | 0.000 |
| 要点覆盖率 ↑ | 0.435 | 0.608 | 0.600 | 0.640 | 0.608 | 0.003 |
| 格式合规率 ↑ | 0.000 | 0.691 | **0.850** | 0.539 | 0.230 | 0.000 |
| 术语准确率 ↑ | 0.853 | 0.882 | 0.824 | 0.882 | 0.882 | 0.000 |
| 拒答准确率 ↑ | 0.000 | 0.716 | **0.890** | 0.126 | 0.716 | 0.000 |
| 误拒率 ↓ | 0.000 | 0.064 | 0.057 | 0.018 | 0.064 | 0.000 |
| **综合分** ↑ | **0.447** | **0.814** | **0.836** | 0.748 | 0.584 | 0.077 |

三个结论：

1. **引用与拒答能力来自数据设计**：去掉拒答样本，拒答准确率从 0.716 掉到 0.126；去掉引用约束，引用指标几乎归零。
2. **没有检索就会编**：无检索消融的幻觉率 0.680，说明“忠实”不是模型自带属性，而是检索 + 约束的结果。
3. **LoRA 在格式/拒答上明显更强**：格式 0.850 vs DPH 0.691，拒答 0.890 vs 0.716；忠实性/术语略低
   （0.889/0.824），幻觉率 0.6%，属于生成式模型的正常代价。

评测口径、测试集构成、人工精评协议见 [`docs/评测说明.md`](docs/评测说明.md)。

---

## 七、Demo 与 2.9 集成

### 7.1 Demo 控制台

```bash
python run_demo.py
# http://127.0.0.1:8802
```

- 同一问题对比 Baseline / 微调模型的回答；
- 展示检索片段、引用编号回链、格式标签、Gate 概率、OOV/覆盖率特征；
- 数据集与评测结果页签；
- 截图见 `screenshots/01~08`。

### 7.2 零改造接入 2.9

Demo 服务同时是 **OpenAI 兼容服务**（`/v1/chat/completions`，支持 SSE）。2.9 只需：

```text
provider = openai
base_url = http://127.0.0.1:8802/v1
api_key  = any
model    = kb-finetuned-dph
```

**为什么不用改 2.9 的代码**：2.9 的 `llm.build_messages` 已经把检索片段拼成带 `[n]` 的 user 消息；
本服务反解这段消息 → 走同一套 DPH 推理 → 按 OpenAI 协议流式返回。2.9 的检索、四维鉴权裁剪、
SSE、引用卡片全部不动。鉴权仍发生在 Prompt 拼接前，**微调不替代权限控制**。

一键验证脚本：

```bash
python integration/check_integration.py \
    --kb http://127.0.0.1:8803 --llm http://127.0.0.1:8802/v1
```

真实接入截图：`screenshots/09-29-chat.png`、`screenshots/10-29-config.png`。

---

## 八、测试

```bash
python -m unittest discover -s tests -v
```

覆盖：文本/数字事实、混合检索、4 类格式组装、拒答/冲突/部分可答、数据集结构/切分泄漏、
策略模型推理、评测指标判定、OpenAI 兼容接口。

---

## 九、已知限制

| 限制 | 说明 / 改进方向 |
| --- | --- |
| LoRA 规模有限 | 只跑 1.5B、2 epoch；7B/更多数据可能进一步提升，但显存/时间成本更高 |
| 格式合规率 0.850（LoRA） | troubleshoot 仅 0.321，仍是主要短板；可补故障类格式指令数据 |
| 要点覆盖率 0.600 | 生成式答案会漏要点；可加“要点清单”监督或更细的 coverage 奖励 |
| 弱相关拒答 0.905（LoRA） | 已明显改善，但部分可答仅 0.40，仍是下一轮重点 |
| 模板同源 | 训练/测试共享问法模板族（文档级切分不泄漏内容）；更严格评测应留出模板 |
| 扫描件 PDF | 沿用 2.9 的能力边界；4.1 数据只使用可解析文本 |

---

## 十、数据合规

- 30 篇制度文档均为 2.9 项目中的**虚构内容**（公司名“星海数字科技”，编号/金额/账号均为占位）；
- 冲突样本中的“2022/2023 旧版”片段为**明确标注的构造数据**，仅用于训练模型识别版本冲突；
- 不包含任何真实企业、个人信息、真实业务数据。

---

## 十一、文档索引

| 文档 | 内容 |
| --- | --- |
| [`docs/微调报告.md`](docs/微调报告.md) | 7 节微调报告（数据→训练→评测→案例→失败→结论） |
| [`docs/数据构造说明.md`](docs/数据构造说明.md) | 主题规格、负样本、质检、切分 |
| [`docs/评测说明.md`](docs/评测说明.md) | 指标定义、判定规则、测试集、人工精评 |
| [`docs/验收对照表.md`](docs/验收对照表.md) | 4.1.1~4.1.5 逐条对照与证据 |
| [`docs/术语表.md`](docs/术语表.md) | 23 个标准术语与口语别称 |
| [`docs/演示脚本.md`](docs/演示脚本.md) | 6 分钟录屏脚本 |
| [`docs/答辩要点.md`](docs/答辩要点.md) | 高频问题、风险与降级 |
| [`docs/项目全流程与知识点详解.md`](docs/项目全流程与知识点详解.md) | 原理、易错点与代码地图 |
| `report/微调报告.html` | 自包含可视化报告（内联 SVG 图表） |
