"""
AI Agent — 能自己调工具的自主助手
项目 2：阶段 B — Agent Loop 从零实现
"""

import os, json, datetime
from openai import OpenAI

ai = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============================================================
# 第1步：定义工具
# ============================================================

def get_current_time():
    """返回当前日期和时间"""
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

def calculate(expression):
    """计算数学表达式，返回结果"""
    try:
        # 安全计算（只允许数字和运算符）
        allowed = set("0123456789+-*/().%^ ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"

def save_note(filename, content):
    """保存笔记到 Obsidian Vault（AI学习/agent_notes/）"""
    dir_path = r"D:\ObsidianVault\AI学习\agent_notes"
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"笔记已保存到 {filepath}"

def read_file(path):
    """读取任意文件内容。参数 path 是文件路径"""
    # 如果是相对路径，从 Vault 根目录找
    if not os.path.isabs(path):
        path = os.path.join(r"D:\ObsidianVault", path)
    if not os.path.exists(path):
        return f"错误：文件不存在 —— {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 2000:
        content = content[:2000] + "\n...(已截断，文件太长)"
    return content

def list_files(directory):
    """列出目录下的文件。参数 directory 是目录路径"""
    if not os.path.isabs(directory):
        directory = os.path.join(r"D:\ObsidianVault", directory)
    if not os.path.exists(directory):
        return f"错误：目录不存在 —— {directory}"
    files = os.listdir(directory)
    return "\n".join(f"  {f}" for f in sorted(files))

def search_notes(query):
    """搜索 Obsidian 笔记（RAG 语义搜索）。参数 query 是搜索问题"""
    # 尝试连接 ChromaDB
    import chromadb
    from chromadb.utils import embedding_functions
    chroma_path = r"D:\ObsidianVault\AI学习\chroma_db"
    if not os.path.exists(chroma_path):
        return "错误：RAG 知识库还没构建，先运行 07-rag-索引构建.py"
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        ef = embedding_functions.DefaultEmbeddingFunction()
        cli = chromadb.PersistentClient(path=chroma_path)
        col = cli.get_collection(name="obsidian_notes", embedding_function=ef)
        results = col.query(query_texts=[query], n_results=3)
        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            parts.append(f"[{meta['source']}]\n{doc[:300]}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"搜索笔记时出错：{e}"

# 工具注册表
TOOLS = {
    "get_current_time": {"func": get_current_time, "desc": "获取当前日期和时间，不需要参数"},
    "calculate": {"func": calculate, "desc": "计算数学表达式，参数 expression 是数学表达式字符串，如 '3*4+2'"},
    "save_note": {"func": save_note, "desc": "保存笔记。参数 filename 文件名, content 正文"},
    "read_file": {"func": read_file, "desc": "读取文件内容。参数 path 是文件路径（可以是相对路径，从Vault根目录找）"},
    "list_files": {"func": list_files, "desc": "列出目录下的文件。参数 directory 是目录路径"},
    "search_notes": {"func": search_notes, "desc": "搜索 Obsidian 知识库中的笔记。参数 query 是搜索的问题"},
}

# ============================================================
# 第2步：把工具描述转成 LLM 能理解的格式
# ============================================================

def build_tools_prompt():
    """生成工具列表的 Prompt"""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['desc']}")
    return "\n".join(lines)

# ============================================================
# 第3步：Agent 核心循环
# ============================================================

SYSTEM_PROMPT = """你是一个自主 AI Agent。你可以执行以下操作：

1. **思考**：分析用户的需求
2. **行动**：调用下面的工具来获取信息或执行任务
3. **观察**：收到工具返回的结果
4. **回答**：根据观察结果回复用户

## 可用工具：
{tools}

## 调用工具的格式（严格遵守）：
当你需要调用工具时，**只输出一行 JSON，不要输出任何其他文字**：
{{"tool": "工具名", "args": {{"参数名": "参数值"}}}}

如果你不需要再调用工具了，直接输出最终答案给用户。

## 规则：
- 每次只调用一个工具
- 调工具时只输出 JSON，绝对不要在前面加"好的"、"我来"等文字
- 工具返回结果后，你再决定下一步
- 能直接回答的不要调工具"""


def agent_loop(user_input, max_steps=5):
    """
    Agent 主循环：
      思考 → 调工具 → 观察结果 → 再思考 → ... → 回答
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools=build_tools_prompt())},
        {"role": "user", "content": user_input},
    ]

    print(f"\n{'='*60}")
    print(f"👤 用户：{user_input}")
    print(f"{'='*60}")

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}：LLM 思考中 ---")

        # 调 LLM
        resp = ai.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        print(f"🤖 LLM 输出：{content[:150]}{'...' if len(content) > 150 else ''}")

        # 尝试从输出中提取 JSON 工具调用（可能在"好的，我先..."之后）
        import re
        tool_call = None
        match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\s*\}', content)
        if not match:
            # 试试更宽松的匹配：找任意 {...} 包含 tool 的
            match = re.search(r'\{.*?"tool".*?\}', content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if "tool" in parsed:
                    tool_call = parsed
            except json.JSONDecodeError:
                pass

        if tool_call is None:
            # 不是工具调用 → 最终回答
            print(f"\n{'='*60}")
            print(f"✅ Agent 最终回答：\n{content}")
            print(f"{'='*60}")
            return content

        # 执行工具
        tool_name = tool_call["tool"]
        tool_args = tool_call.get("args", {})

        if tool_name not in TOOLS:
            result = f"错误：没有叫 '{tool_name}' 的工具"
        else:
            print(f"🔧 调用工具：{tool_name}({tool_args})")
            try:
                result = TOOLS[tool_name]["func"](**tool_args)
            except TypeError:
                result = TOOLS[tool_name]["func"]()
            print(f"📋 工具返回：{result}")

        # 把工具结果喂回 LLM
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": f"工具 {tool_name} 返回的结果：{result}\n\n请根据这个结果继续。如果信息够了就直接回答，否则继续调工具。"
        })

    print("\n⚠️ 达到最大步数限制，Agent 停止。")
    return None


# ============================================================
# 第4步：交互入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖  AI Agent 已启动（6 个工具）")
    print("   试试这些：")
    print("   ① 帮我看看AI学习文件夹里有什么文件？")
    print("   ② 打开 AI_学习路线图.md，告诉我路线有几阶段")
    print("   ③ 搜索笔记：temperature 应该怎么用")
    print("   ④ 现在几点？帮我算 365×24，保存到笔记里")
    print("   输入 quit 退出")
    print("=" * 60)

    while True:
        q = input("\n👤 你: ").strip()
        if q.lower() == "quit":
            print("👋 拜拜！")
            break
        if not q:
            continue
        agent_loop(q)
