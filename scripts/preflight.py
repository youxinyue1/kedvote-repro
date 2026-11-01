# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""跑批链路预检：小模型问一题，验证推理链路正常"""
import json
import urllib.request

payload = {
    "model": "deepseek-r1:1.5b",
    "messages": [{"role": "user", "content": "回答两个字：人参的性味?"}],
    "temperature": 0.0,
    "max_tokens": 2048,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:11434/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    c = json.loads(r.read())["choices"][0]["message"]["content"]
# 剥 R1 思维链（用变量拼接避免字面量被传输层误处理）
tag = "<" + "/think" + ">"
if tag in c:
    c = c.split(tag, 1)[1]
print("[预检] deepseek-r1:1.5b =>", (c or "")[:80])
