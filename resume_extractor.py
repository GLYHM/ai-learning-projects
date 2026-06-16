"""
实战项目：简历信息提取器
技术栈：Few-shot + JSON 输出 + Chain-of-Thought
模块: AI学习 - 阶段A 第2周 - 实战练习
"""

import os, json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============================================================
# 核心函数：提取简历信息
# ============================================================

def extract_resume(resume_text):
    """
    输入一段简历文本，输出结构化的 JSON。
    用了 Few-shot + Chain-of-Thought + JSON 约束。
    """

    # 给 AI 一个详细的例子（Few-shot）
    example_input = """
    李明，男，27岁。2022年毕业于浙江大学计算机科学专业，本科学历。
    2022-2024年在阿里巴巴担任前端工程师，负责淘宝首页改版，使用React和TypeScript。
    2024年至今在字节跳动担任高级前端工程师，负责抖音Web端性能优化。
    技能：React, TypeScript, JavaScript, Webpack, Node.js。
    期望职位：前端架构师。期望薪资：40K-50K。
    电话：13812345678，邮箱：liming@email.com。
    """

    example_output = """
    {
      "name": "李明",
      "gender": "男",
      "age": 27,
      "phone": "13812345678",
      "email": "liming@email.com",
      "education": [
        {"school": "浙江大学", "major": "计算机科学", "degree": "本科", "year": 2022}
      ],
      "work_experience": [
        {"company": "阿里巴巴", "title": "前端工程师", "period": "2022-2024", "responsibilities": ["淘宝首页改版，React+TypeScript"]},
        {"company": "字节跳动", "title": "高级前端工程师", "period": "2024-至今", "responsibilities": ["抖音Web端性能优化"]}
      ],
      "skills": ["React", "TypeScript", "JavaScript", "Webpack", "Node.js"],
      "target_position": "前端架构师",
      "expected_salary": "40K-50K"
    }
    """

    prompt = f"""你是一个专业的简历解析引擎。请从下面的简历文本中提取信息，输出 JSON。

=== 示例 ===
输入：{example_input}
输出：{example_output}

=== 现在解析以下简历 ===
{resume_text}

请严格按示例格式输出 JSON。如果某项信息没有提到，对应的值填 null。只输出 JSON，不要解释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "你是一个精确的简历解析引擎，只输出JSON，不输出其他内容。"},
                  {"role": "user", "content": prompt}],
        temperature=0,   # 提取任务，稳定性优先
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()
    # 清理可能的 markdown 代码块标记
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


# ============================================================
# 测试：两段真实的简历
# ============================================================

if __name__ == "__main__":
    # 测试简历1：详细的
    resume1 = """
    王小芳，女，30岁。2015年毕业于复旦大学新闻学专业，硕士学历。
    2015-2018年在腾讯新闻担任内容编辑，负责时事新闻采编和内容审核。
    2018-2023年在澎湃新闻担任高级编辑，负责深度报道策划和团队管理。
    技能：新闻采编，内容策划，团队管理，数据分析（Excel、SQL），Axure原型设计。
    期望职位：内容总监。期望薪资：30K以上。
    电话：13900001111。
    """

    print("📄 简历 1：")
    result1 = extract_resume(resume1)
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    # 测试简历2：信息不完整的
    resume2 = """
    我是张工，做了5年Java后端，熟悉Spring Boot、MySQL、Redis。
    在美团和滴滴都干过。期望找个35K左右的架构岗。邮箱是zhanggong@test.com。
    """

    print("\n📄 简历 2（信息不全）：")
    result2 = extract_resume(resume2)
    print(json.dumps(result2, ensure_ascii=False, indent=2))

    print("\n✅ 两份简历全部结构化提取完成！可以直接存入数据库或 Excel。")
