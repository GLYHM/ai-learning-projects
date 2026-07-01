"""
LangChain Chain 示例 — 串联多个 LLM 调用

你之前没用过 Chain，这是 LangChain 独有的概念：
  Chain = 多个步骤串成一条流水线

示例：题目 → 解题 → 验证，三个 LLM 调用串联

运行方式:
  set DEEPSEEK_API_KEY=你的Key
  python 13-langchain-chain.py
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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==== 示例1: 简单 Chain — 两步流水线 ====
print("=" * 50)
print("示例1: 题目 → 解题")
print("=" * 50)

generate_prompt = ChatPromptTemplate.from_template(
    "解这道数学题，写出步骤：{question}"
)

solver = generate_prompt | llm | StrOutputParser()

question = "一个水池，进水管 3 小时注满，出水管 5 小时放空。两管同时开，几小时注满？"
answer = solver.invoke({"question": question})
print(f"题: {question[:50]}...")
print(f"解: {answer[:300]}")

# ==== 示例2: 多步 Chain — 解题 → 验证 ====
print("\n" + "=" * 50)
print("示例2: 解题 → 验证 → 最终答案")
print("=" * 50)

verify_prompt = ChatPromptTemplate.from_template(
    """检查以下解题过程是否正确。如果正确，说"正确"。如果错误，指出错在哪里并给出正确答案。

原题：{question}

解题过程：{solution}

评审："""
)

# 链式串联：先解题，把结果传入验证
solve_and_verify = (
    {"question": RunnablePassthrough(), "solution": solver}
    | verify_prompt
    | llm
    | StrOutputParser()
)

result = solve_and_verify.invoke(question)
print(f"验证结果: {result[:300]}")


# ==== 示例3: 你的实战场景 — 笔记生成器 Chain 版 ====
print("\n" + "=" * 50)
print("示例3: 文章 → 提取摘要 → 翻译 → Obsidian 格式")
print("=" * 50)

article = (
    "LangChain is a framework for developing applications powered by "
    "language models. It provides composable tools and modular components, "
    "enabling developers to build complex LLM applications with just a few lines of code."
)

extract_prompt = ChatPromptTemplate.from_template(
    "提取以下文章的要点列表：\n{text}"
)

translate_prompt = ChatPromptTemplate.from_template(
    "把以下要点翻译成中文：\n{points}"
)

format_prompt = ChatPromptTemplate.from_template(
    "把以下内容格式化成 Markdown 笔记，加标题和标签：\n{translated}"
)

# 三步骤 Chain
pipeline = (
    {"text": RunnablePassthrough()}
    | extract_prompt | llm | StrOutputParser()
    | (lambda x: {"points": x})
    | translate_prompt | llm | StrOutputParser()
    | (lambda x: {"translated": x})
    | format_prompt | llm | StrOutputParser()
)

result = pipeline.invoke(article)
print(result[:400])

print("\n" + "=" * 50)
print("三个示例运行完毕。")
print("核心概念: | 管道符把多个 LLM 调用串成流水线")
print("=" * 50)
