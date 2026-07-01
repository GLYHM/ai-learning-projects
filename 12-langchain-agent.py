"""
LangChain Agent 示例 — 对照手写版 09-agent 理解

你手写的 Agent 做了什么：
  1. 定义 6 个工具函数
  2. LLM 输出 JSON {"tool": "xxx", "args": {...}}
  3. 正则提取 JSON + 容错
  4. 执行工具 → 结果喂回 LLM → 循环直到输出最终答案

LangChain 帮你做了什么：
  1. @tool 装饰器把任意函数变成工具
  2. create_tool_calling_agent 自动处理工具调用循环
  3. AgentExecutor 自动处理 JSON 解析、错误重试、步数限制
  4. 不需要自己写正则、不需要管循环终止条件

关键对比:
  你手写 09-agent.py:  150 行核心 Agent 循环代码
  LangChain Agent:      10 行搞定同样的事

运行方式:
  set DEEPSEEK_API_KEY=你的Key
  python 12-langchain-agent.py
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ==== 1. 定义工具（替代你手写的 6 个函数） ====
from langchain_core.tools import tool
import datetime, math

@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 '2+3*4'"""
    allowed = set("0123456789+-*/() .%")
    cleaned = "".join(c for c in expression if c in allowed)
    try:
        result = eval(cleaned)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def save_note(title: str, content: str) -> str:
    """保存一篇笔记到 Obsidian Vault"""
    vault = os.path.expanduser("~/Documents/ObsidianVault/AI学习")
    filepath = os.path.join(vault, f"{title}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}\n")
    return f"笔记已保存到 {filepath}"

@tool
def search_web(query: str) -> str:
    """搜索互联网获取实时信息（模拟）"""
    # 实际项目中接入搜索 API
    return f"关于「{query}」的搜索结果（模拟）：这是一个演示工具。"

tools = [get_current_time, calculate, save_note, search_web]

# ==== 2. 创建 Agent ====
from langchain.agents import create_agent

agent = create_agent(llm, tools)

# ==== 3. 测试 ====
print("=" * 50)
print("LangChain Agent 测试")
print("=" * 50)

tasks = [
    "现在几点？",
    "计算 123 + 456 * 2",
    "计算 (100 + 200) / 3，然后把结果告诉我",
]

for task in tasks:
    print(f"\n用户: {task}")
    result = agent.invoke({"messages": [("user", task)]})
    # Agent 返回的消息列表，最后一条是最终回答
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = getattr(msg, "type", "?")
            content = str(msg.content)[:200]
            if role == "ai":
                print(f"Agent: {content}")
