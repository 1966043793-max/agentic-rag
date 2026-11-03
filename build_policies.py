"""一次性工具：把抓取到的两部法律全文清洗并生成 policies.json。

用法：python build_policies.py
之后可删除本脚本和 consumer_law.txt / ecommerce_law.txt。
"""
import json
import re
from pathlib import Path

ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百]+条")


def clean_law(path, footer_markers=("相关链接", "责任编辑", "【E-mail")):
    text = Path(path).read_text(encoding="utf-8")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 1) 找到第一条正文的位置
    start = next(i for i, l in enumerate(lines) if ARTICLE_RE.match(l))
    # 2) 往前回溯到最近的"第X章"，把章节标题也带上（提供上下文）
    for j in range(start - 1, -1, -1):
        if re.match(r"^第[一二三四五六七八九十百]+章", lines[j]):
            start = j
            break

    # 3) 找到正文终点（页脚噪声之前）
    end = len(lines)
    for i in range(start, len(lines)):
        if any(lines[i].startswith(m) for m in footer_markers):
            end = i
            break
    body = lines[start:end]

    # 4) 分组：遇到"第X条 / 第X章"就开新段，其余续行用空格接上
    paragraphs = []
    for l in body:
        if re.match(r"^第[一二三四五六七八九十百]+[章条]", l):
            paragraphs.append(l)
        elif paragraphs:
            paragraphs[-1] += " " + l
        else:
            paragraphs.append(l)

    return "\n\n".join(paragraphs)


consumer = clean_law("consumer_law.txt")
ecommerce = clean_law("ecommerce_law.txt")

docs = [
    {"id": "DOC-consumer-rights", "title": "中华人民共和国消费者权益保护法", "content": consumer},
    {"id": "DOC-ecommerce", "title": "中华人民共和国电子商务法", "content": ecommerce},
]

out = Path("data/seed/policies.json")
out.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"消费者权益保护法：{len(consumer)} 字符")
print(f"电子商务法：{len(ecommerce)} 字符")
print(f"已写入 {out}")
