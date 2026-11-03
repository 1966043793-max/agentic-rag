"""一次性工具：把「每部法律一个文档」拆成「每条条文一个文档」。

用法：python split_articles.py
结果：policies.json 从 2 个 doc 变成 ~152 个 doc（每条一个），
      id 形如 crpl-25（消保法第25条）/ ecl-49（电商法第49条）。
"""
import json
import re
from pathlib import Path

SRC = Path("data/seed/policies.json")

LAW_PREFIX = {
    "DOC-consumer-rights": "crpl",
    "DOC-ecommerce": "ecl",
}
LAW_NAME = {
    "DOC-consumer-rights": "消费者权益保护法",
    "DOC-ecommerce": "电子商务法",
}

data = json.loads(SRC.read_text(encoding="utf-8"))

articles = []
for doc in data:
    prefix = LAW_PREFIX[doc["id"]]
    law_name = LAW_NAME[doc["id"]]
    segments = [s.strip() for s in doc["content"].split("\n\n") if s.strip()]
    counter = 0
    for seg in segments:
        m = re.match(r"^(第[一二三四五六七八九十百]+条)", seg)
        if not m:  # 跳过章节标题（第一章、第二章…）
            continue
        counter += 1
        article_num = m.group(1)  # 如「第二十五条」
        articles.append({
            "id": f"{prefix}-{counter:02d}",
            "title": f"{law_name} {article_num}",
            "content": seg,
        })

SRC.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"拆分完成：共 {len(articles)} 条")
