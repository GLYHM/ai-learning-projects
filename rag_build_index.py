"""
RAG 知识库助手 — 第1步：构建索引
使用 HuggingFace 镜像，国内可访问
"""

import os, shutil

# 🔑 关键：用国内镜像，不翻墙下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from chromadb.utils import embedding_functions

VAULT_PATH = r"D:\ObsidianVault"
CHROMA_PATH = r"D:\ObsidianVault\AI学习\chroma_db"

# ============================================================
# 第1步：加载嵌入模型（走 HF 镜像）
# ============================================================
print("📦 正在下载/加载嵌入模型（走国内镜像，首次约 80MB）...")
embedder = embedding_functions.DefaultEmbeddingFunction()
print("✅ 模型就绪\n")

# ============================================================
# 第2步：把笔记切成小块
# ============================================================
def load_and_split(vault):
    chunks = []
    for root, _, files in os.walk(vault):
        if ".obsidian" in root or ".claude" in root or "chroma_db" in root:
            continue
        for f in files:
            if not f.endswith(".md"):
                continue
            path = os.path.join(root, f)
            content = open(path, encoding="utf-8").read()
            if len(content) < 50:
                continue
            sections = content.split("\n## ")
            for i, sec in enumerate(sections):
                sec = sec.strip()
                if len(sec) < 30:
                    continue
                chunks.append({
                    "text": sec,
                    "source": os.path.relpath(path, VAULT_PATH),
                })
    return chunks

print("🔍 正在扫描笔记...")
chunks = load_and_split(VAULT_PATH)
print(f"✅ 共 {len(chunks)} 个文本块，来自 {len(set(c['source'] for c in chunks))} 个文件\n")

# ============================================================
# 第3步：存入 ChromaDB（自动生成语义向量）
# ============================================================
print("💾 正在构建向量数据库...")
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.create_collection(
    name="obsidian_notes",
    embedding_function=embedder,
)

batch_size = 50
for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]
    collection.add(
        ids=[str(start + i) for i in range(len(batch))],
        documents=[c["text"] for c in batch],
        metadatas=[{"source": c["source"]} for c in batch],
    )
    print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)} 块")

print(f"\n✅ 索引构建完成！{len(chunks)} 个文本块已存入 {CHROMA_PATH}")
print("🎉 下一步：运行 08-rag-查询.py 开始对话！")
