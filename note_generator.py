"""
项目3：AI 学习笔记生成器
文章 → 要点提取 → RAG 搜相关笔记 → 生成 Obsidian 笔记 → 自动保存
"""

import os, json, datetime, re
from openai import OpenAI
import requests

ai = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

NOTE_DIR = r"D:\ObsidianVault\AI学习"


# ============================================================
# 工具：获取网页内容
# ============================================================
def fetch_url(url):
    """抓取网页标题和正文文本"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        return "", f"❌ 请求失败：{e}"

    html = resp.text

    # 提取标题
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
    page_title = title_match.group(1).strip() if title_match else url

    # 简单提取文本（去掉所有 HTML 标签和 script/style）
    for tag in ["script", "style", "nav", "footer", "header"]:
        html = re.sub(f"<{tag}.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    # 截断，防止太长
    if len(text) > 6000:
        text = text[:6000] + "\n\n...（文章过长，已截断前 6000 字）"

    return page_title, text


# ============================================================
# 工具：RAG 搜索已有笔记
# ============================================================
def search_related_notes(topic, top_k=3):
    """搜索 Vault 中跟这个主题相关的笔记，返回 [[链接]] 列表"""
    import chromadb
    from chromadb.utils import embedding_functions

    chroma_path = r"D:\ObsidianVault\AI学习\chroma_db"
    if not os.path.exists(chroma_path):
        return "（知识库未构建）"

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    ef = embedding_functions.DefaultEmbeddingFunction()
    cli = chromadb.PersistentClient(path=chroma_path)
    col = cli.get_collection(name="obsidian_notes", embedding_function=ef)
    results = col.query(query_texts=[topic], n_results=top_k)

    links = []
    seen = set()
    for meta in results["metadatas"][0]:
        src = meta["source"]
        if src in seen:
            continue
        seen.add(src)
        # 转成 Obsidian 内部链接
        name = src.replace("\\", "/").replace(".md", "")
        links.append(f"  - [[{name}]]")
    return "\n".join(links) if links else "（暂无相关笔记）"


# ============================================================
# 核心：文章 → Obsidian 笔记
# ============================================================
def article_to_note(article_text, source_name=""):
    """输入一篇文章，返回完整的 Obsidian 笔记 Markdown"""

    today = datetime.date.today().strftime("%Y-%m-%d")

    # ---- Step 1：LLM 提取结构化信息 ----
    extract_prompt = f"""你是一个技术笔记整理专家。阅读以下文章，提取关键信息。

文章来源：{source_name}

文章内容：
{article_text[:4000]}

请严格按以下 JSON 格式输出（只输出 JSON，别的不输出）：
{{
  "title": "笔记标题（简洁，15字以内）",
  "tags": ["标签1", "标签2", "标签3"],
  "summary": "200字以内的核心摘要，用中文",
  "key_points": ["要点1", "要点2", "要点3", "要点4", "要点5"],
  "concepts": [
    {{"name": "概念名1", "explain": "一句话解释"}},
    {{"name": "概念名2", "explain": "一句话解释"}}
  ],
  "code_example": "如果文章有代码示例，提取到这里（没有填 null）",
  "my_thought": "一段话：这篇文章和我已有的知识有什么关联，对我的价值是什么"
}}"""

    resp = ai.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": extract_prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    raw = resp.choices[0].message.content.strip()
    # 清理可能的 markdown 包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3]

    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return f"❌ JSON 解析失败，LLM 返回：\n{raw[:300]}"

    # ---- Step 2：RAG 搜索相关笔记 ----
    related = search_related_notes(info["title"])

    # ---- Step 3：生成 Obsidian 格式笔记 ----
    tags_str = ", ".join(info["tags"])
    key_points_str = "\n".join(f"- {p}" for p in info["key_points"])
    concepts_str = "\n".join(
        f"- **{c['name']}**：{c['explain']}" for c in info["concepts"]
    )

    code_block = ""
    if info.get("code_example") and info["code_example"] != "null":
        code_block = f"\n## 代码示例\n\n```\n{info['code_example']}\n```\n"

    note = f"""---
tags: [{tags_str}]
created: {today}
source: {source_name}
---

# {info['title']}

## 核心摘要

{info['summary']}

## 关键要点

{key_points_str}

## 核心概念

{concepts_str}
{code_block}
## 我的思考

{info['my_thought']}

## 相关笔记

{related}
"""
    return note, info["title"]


# ============================================================
# 交互入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📝  AI 学习笔记生成器")
    print("   输入 1：粘贴文本")
    print("   输入 2：读取本地文件")
    print("   输入 3：输入网页链接")
    print("   输入 quit 退出")
    print("=" * 60)

    while True:
        choice = input("\n👉 选择: ").strip()

        if choice.lower() == "quit":
            print("👋 拜拜！")
            break

        if choice == "1":
            print("📄 请粘贴文章内容（输入 END 结束）：")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            article = "\n".join(lines)
            source = input("📎 来源（可选，回车跳过）: ").strip() or "手动输入"

        elif choice == "2":
            path = input("📂 文件路径: ").strip()
            if not os.path.isabs(path):
                path = os.path.join(r"D:\ObsidianVault", path)
            if not os.path.exists(path):
                print(f"❌ 文件不存在: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                article = f.read()
            source = os.path.basename(path)

        elif choice == "3":
            url = input("🔗 网页链接: ").strip()
            if not url.startswith("http"):
                print("❌ 请输入完整链接（以 http 开头）")
                continue
            print("🌐 抓取网页中...")
            source, article = fetch_url(url)
            if not article:
                print(article)
                continue
            print(f"   标题: {source} | 正文: {len(article)} 字")

        else:
            continue

        if len(article.strip()) < 50:
            print("❌ 内容太短，至少 50 字")
            continue

        print("\n⚙️  处理中...")
        note, title = article_to_note(article, source)

        # 保存
        filename = title.replace("/", "-").replace(":", " -") + ".md"
        filepath = os.path.join(NOTE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(note)

        print(f"\n{'='*60}")
        print(f"✅ 笔记已生成！")
        print(f"   标题：{title}")
        print(f"   保存到：{filepath}")
        print(f"{'='*60}")
        print(f"\n📖 预览（前 500 字）：\n{note[:500]}...")
