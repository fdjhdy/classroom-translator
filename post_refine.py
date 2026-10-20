# -*- coding: utf-8 -*-
"""
课后精翻：拿上课时存的整段 WAV 录音，用高质量参数重新识别 + 带上下文翻译，
产出一份干净的课堂笔记（Markdown）。

用法（在主机上，模型首次会下载 large-v3 约 3G；Key 从环境变量 TRANSLATE_API_KEY 读取）：
    python post_refine.py records\20260914_1930_宏观经济学.wav ^
        --subject "宏观经济学"

用硅基流动加：--base-url https://api.siliconflow.cn/v1 --translate-model deepseek-ai/DeepSeek-V3
"""
import argparse
import os
import time
import wave

import numpy as np
from faster_whisper import WhisperModel
from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("wav", help="上课时存的 wav 录音路径")
p.add_argument("--key", default=os.environ.get("TRANSLATE_API_KEY", ""),
               help="翻译 API 的 key；不传则从环境变量 TRANSLATE_API_KEY 读取")
p.add_argument("--base-url", default="https://api.deepseek.com")
p.add_argument("--translate-model", default="deepseek-chat")
p.add_argument("--subject", default="", help="课程名，喂给翻译模型做上下文")
p.add_argument("--glossary-file", default="", help="术语表 txt（每行：俄语=中文）")
p.add_argument("--model", default="large-v3", help="ASR 模型，课后不赶时间，用 large-v3")
p.add_argument("--device", default="cuda")
p.add_argument("--compute-type", default="int8_float16")
args = p.parse_args()

if not args.key:
    p.error("缺少翻译 API Key：用 --key 传入，或先 set TRANSLATE_API_KEY=sk-...")

glossary = ""
if args.glossary_file:
    with open(args.glossary_file, encoding="utf-8") as f:
        glossary = f.read().strip()

# ---------------- 读录音 ----------------
with wave.open(args.wav, "rb") as w:
    if w.getframerate() != 16000 or w.getsampwidth() != 2 or w.getnchannels() != 1:
        raise SystemExit("wav 格式不符（需要 16kHz 单声道 16bit）——只支持本系统录的文件")
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
duration_min = len(audio) / 16000 / 60
print(f"[录音] {duration_min:.1f} 分钟")

# ---------------- 高质量识别 ----------------
print(f"[加载模型] {args.model} ...")
model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
print("[识别] 开始（一节课约几分钟，耐心等）...")
segments, info = model.transcribe(
    audio, language="ru", beam_size=5, vad_filter=True,
    condition_on_previous_text=True,   # 课后批量跑可以开，连贯性更好
)
segments = list(segments)
print(f"[识别] 完成，共 {len(segments)} 句")

# ---------------- 带上下文翻译 ----------------
client = OpenAI(api_key=args.key, base_url=args.base_url, timeout=60.0)
system = (
    f"你是俄语翻译专家。以下是俄罗斯大学「{args.subject or '专业'}」课程的课堂录音转写，"
    "逐句翻译成准确、通顺的简体中文。转写可能有识别错误，请结合上下文合理推断；"
    "术语保持全文一致。只输出译文本身。"
)
if glossary:
    system += f"\n术语对照表（优先采用）：\n{glossary}"


def translate(text, context):
    user = f"【前文，仅供理解】\n{context}\n\n【要翻译的句子】\n{text}" if context else text
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=args.translate_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2, max_tokens=800)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [重试 {attempt + 1}/3] {e}")
            time.sleep(2 * (attempt + 1))
    return f"[翻译失败：{e}]"


def fmt_ts(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


out_path = os.path.splitext(args.wav)[0] + "_精翻.md"
history = []
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"# {args.subject or '课堂'} 精翻笔记\n\n"
            f"来源录音：{os.path.basename(args.wav)}（{duration_min:.0f} 分钟）\n\n---\n\n")
    for i, s in enumerate(segments, 1):
        ru = s.text.strip()
        if not ru:
            continue
        ctx = "\n".join(history[-4:])
        zh = translate(ru, ctx)
        history.append(f"{ru} / {zh}")
        f.write(f"**[{fmt_ts(s.start)}]** {ru}\n\n{zh}\n\n---\n\n")
        f.flush()
        if i % 10 == 0 or i == len(segments):
            print(f"[翻译] {i}/{len(segments)}")

print(f"\n[完成] 笔记已保存: {out_path}")
print("下一步：把这个 md 文件发给 AI，让它出知识点总结 + 术语表，回填 glossary.txt")
