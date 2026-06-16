"""
RAG 知识库助手 — 第2步：语义问答 + 多轮对话
向量语义搜索 + 上下文记忆 → DeepSeek 回答
"""

import os, sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = r"D:\ObsidianVault\AI学习\chroma_db"

if not os.path.exists(CHROMA_PATH):
    print("❌ 先构建索引: python 07-rag-索引构建.py")
    sys.exit(1)

# ============================================================
# 加载
# ============================================================
print("📦 加载中...")
embedder = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(
    name="obsidian_notes",
    embedding_function=embedder,
)
ai = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
print("✅ 就绪！\n")

# ============================================================
# RAG + 多轮对话
# ============================================================

# 对话历史（每轮都带上，AI 不会失忆）
history = []

def rag_chat(question, top_k=3):

    # 1. 语义搜索笔记
    results = collection.query(query_texts=[question], n_results=top_k)
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[来源: {meta['source']}]\n{doc}")
    context = "\n\n---\n\n".join(context_parts) if context_parts else "(无相关笔记)"

    # 2. 把搜索结果加入对话历史
    history.append({
        "role": "user",
        "content": f"（参考笔记）\n{context}\n\n（用户问题）\n{question}"
    })

    # 3. 带全部历史调 AI
    resp = ai.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是知识库助手。根据用户提供的笔记回答问题，有据可查就引用来源，没找到就说没找到，绝不编造。记住之前的对话。"},
        ] + history[-10:],  # 保留最近 10 轮，防止过长
        temperature=0,
        max_tokens=500,
    )

    reply = resp.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


# ============================================================
# 交互
# ============================================================
print("=" * 60)
print("🤖  RAG 知识库助手 v2.0（语义搜索 + 对话记忆）")
print("   试试连续追问：")
print("   ① AI 学习有几个阶段？")
print("   ② 哪个阶段最重要？ ← 不用再提'AI学习'，它记得上下文")
print("   ③ 我该先做什么？")
print("   输入 quit 退出")
print("=" * 60)

while True:
    q = input("\n👤 你: ").strip()
    if q.lower() == "quit":
        print("👋 拜拜！")
        break
    if not q:
        continue
    print("🔍 搜索笔记 + 思考中...")
    answer = rag_chat(q)
    print(f"\n🤖 AI: {answer}")
