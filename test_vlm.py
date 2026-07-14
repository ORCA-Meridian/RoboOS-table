"""
测试 VLM 配置是否可用。
运行：python test_vlm.py
"""
import base64
import httpx
from openai import OpenAI

# API_KEY  = "sk-24c26577d05b4b0598acace7c666aadc"
# API_BASE = "https://llm-yl7qhh3l03qfmpuk.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# MODEL    = "qwen3.7-plus"

API_KEY  = "sk-wsCXaesxBcVxIBaEC79f3aD05fC24560941261D5CeE420Da"
API_BASE = "https://aihubmix.com/v1"
MODEL    = "gpt-4o"

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ── 测试1：纯文字，验证 API Key 和地址是否通 ──────────────────────────
print("=" * 50)
print("测试1：纯文字对话")
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "回复 OK"}],
        # max_tokens=100,
    )
    print("✅ 成功:", repr(resp.choices[0].message.content))
    # print("完整 message:", resp.choices[0].message)
    # print("usage:", resp.usage)
except Exception as e:
    print("❌ 失败:", e)

# ── 测试2：图像输入，验证视觉能力 ────────────────────────────────────
print()
print("=" * 50)
print("测试2：图像输入（用一张网络图片）")
try:
    # 下载一张小图做测试
    # img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    img_url = "https://img-s.msn.cn/tenant/amp/entityid/AA26KvOL.img?w=640&h=820&m=6"
    img_bytes = httpx.get(img_url, timeout=10).content
    b64 = base64.b64encode(img_bytes).decode()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + b64},
                },
                {"type": "text", "text": "这张图里是什么。"},
            ],
        }],
        # max_tokens=300,
        temperature=0.0,
    )
    print("✅ 成功:", repr(resp.choices[0].message.content))
    # print("完整 message:", resp.choices[0].message)
    # print("usage:", resp.usage)
except Exception as e:
    print("❌ 失败:", e)
    print()
    print("如果报 model not found，把 MODEL 改成下面列出的可用模型之一")
    print("运行 list_models.py 查看该端点下可用的模型列表")
