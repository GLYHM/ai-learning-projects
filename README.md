# AI Learning Projects

> 在职自学 AI 应用开发 — 从零到项目落地的完整记录
>
> 每天 1-2 小时，3 个月，6 个可运行的项目

---

## 关于我

制造业 EHS 数智化专员，非计算机专业，从零开始自学 AI 应用开发。目标岗位：**AI 应用开发工程师**。

这些项目是我学习过程的实战产出，不是教程作业——每个项目都解决了一个真实问题。

---

## 项目列表

### 🤖 AI Agent（6 工具自主助手）

**`agent.py`** — 从零实现的 AI Agent 架构

```
用户提问 → Think（LLM决策） → Act（执行工具） → Observe（观察结果） → 回答
```

| 特性 | 实现 |
|------|------|
| 架构 | Think → Act → Observe 循环（手写，不用框架） |
| 工具 | 时间查询、数学计算、文件读写、目录浏览、RAG 笔记搜索、笔记保存 |
| 容错 | 正则提取 JSON（LLM 输出不总是纯 JSON） |
| 步数限制 | 最多 5 步，防止无限循环 |

**核心代码片段：**

```python
for step in range(max_steps):
    content = llm_response(messages)
    tool_call = extract_json(content)      # 正则容错提取
    if tool_call is None:
        return content                      # 最终回答
    result = execute_tool(tool_call)        # 执行工具
    messages.append({"role": "user", "content": f"工具结果：{result}"})
```

---

### 📚 RAG 知识库助手

**`rag_build_index.py`** + **`rag_query.py`** — 给 Obsidian 笔记装上语义搜索

```
笔记(.md) → 按标题切块 → Embedding 向量化 → ChromaDB 存储
                                                ↓
用户提问 → 向量化 → 语义搜索 Top 3 → 拼接 Prompt → LLM 回答
```

| 特性 | 实现 |
|------|------|
| 文本切分 | 按 `##` 标题分块（保留上下文） |
| 向量模型 | all-MiniLM-L6-v2（ONNX 运行时，无需 PyTorch） |
| 向量数据库 | ChromaDB |
| 对话记忆 | 多轮对话历史，最近 10 条 |
| 国内适配 | HuggingFace 镜像 + 清华 pip 源 |

**一个真实对比：**

| 问题 | TF-IDF（关键词） | Embedding（语义） |
|------|:---:|:---:|
| "AI学习有几个阶段" | ❌ 没找到 | ✅ 找到了 |

语义搜索理解"几个阶段"和 "3个阶段" 是同一回事。

---

### 📝 笔记生成器

**`note_generator.py`** — 文章 → AI 提取 → RAG 搜索 → Obsidian 笔记，全自动

```
URL/文本 → LLM 提取结构化信息 → RAG 搜索相关笔记 → 生成 Obsidian 格式 → 自动保存
```

| 特性 | 实现 |
|------|------|
| 输入方式 | 粘贴文本 / 本地文件 / 网页链接 |
| 网页抓取 | requests + 正则清洗 HTML |
| 结构化提取 | Few-shot JSON Prompt（标题、标签、摘要、要点、概念、代码、思考） |
| 关联推荐 | RAG 搜索已有笔记，自动加 `[[链接]]` |
| 输出 | Frontmatter + Markdown，存到 Obsidian Vault |

**技术栈串联：**

```
Few-shot JSON 提取 → RAG 语义搜索 → Obsidian 双向链接
     (06学)           (07/08学)         (Obsidian特性)
```

---

### 🔧 基础工具

| 文件 | 说明 |
|------|------|
| `ask_ai.py` | DeepSeek API 三行封装函数 `ask_ai(prompt, role, temperature)` |
| `resume_extractor.py` | Few-shot + JSON 结构化提取简历信息（工业级模式演示） |

---

## 技术栈

```
语言：Python 3.12
API：DeepSeek（OpenAI 兼容）
模型：deepseek-chat / deepseek-reasoner
向量数据库：ChromaDB + ONNX Embedding（all-MiniLM-L6-v2）
无框架：全部手写，不依赖 LangChain / LlamaIndex
```

## 学习路径

```
阶段 A（4周）：会用 AI ──→ API调用 → Prompt Engineering → 日常融入
阶段 B（8周）：会做产品 ──→ RAG → Agent → 笔记生成器
阶段 C（持续）：会交付   ──→ GitHub建设 → 技术博客 → 面试准备
```

---

## 运行环境

```bash
pip install openai chromadb requests
```

设置环境变量：
```bash
set DEEPSEEK_API_KEY=你的DeepSeek密钥
```

> **注意**：国内运行 ChromaDB 需要设置 HuggingFace 镜像：
> ```python
> os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
> ```

---

## 关键工程踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| PyTorch DLL 加载失败 | 缺少 VC++ 运行库 | `winget install Microsoft.VCRedist` |
| HuggingFace 下载超时 | 国内网络限制 | 使用 `hf-mirror.com` 镜像 |
| ChromaDB ONNX SSL 错误 | 同上 | 改用镜像 + ONNX 替代 PyTorch |
| Agent JSON 解析失败 | LLM 在 JSON 前加了自然语言 | 正则提取 `{...}` + 强化 System Prompt |
| pip 安装超时 | PyPI 被墙 | 清华镜像 `pypi.tuna.tsinghua.edu.cn` |

---

## 下一步

- [ ] LangChain 框架学习
- [ ] FastAPI 把项目部署成 API 服务
- [ ] Docker 容器化
- [ ] 公司内部 EHS 法规助手（真实业务落地）

---

*最后更新：2026-06-16*
