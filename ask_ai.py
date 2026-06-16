"""
封装你的第一个 AI 工具函数
模块: AI学习 - 阶段A 第1周
"""

import os
from openai import OpenAI


def ask_ai(prompt, role="你是一个有用的助手", temperature=0.7):
    """
    问 AI 一个问题，返回回答字符串。

    参数:
        prompt  : 你要问的问题
        role    : 想让 AI 扮演什么角色（默认是通用助手）
        temperature: 0=稳定, 1.5=狂野（默认 0.7）
    返回:
        AI 的回答（字符串）
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=300,
    )
    return resp.choices[0].message.content


# ============================================
# 测试：一行代码就能问 AI！
# ============================================

if __name__ == "__main__":
    # 玩法1：直接问
    print("📝 基础提问：")
    print(ask_ai("Python中列表和字典有什么区别？"))
    print()

    # 玩法2：换角色
    print("🎭 角色扮演（小学老师）：")
    print(ask_ai("什么是深度学习？", role="你是一个小学科学老师，用简单的比喻解释。"))
    print()

    # 玩法3：调温度
    print("🎲 高温度写诗：")
    print(ask_ai("写一首关于代码的爱情诗", temperature=1.2))
