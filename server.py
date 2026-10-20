# -*- coding: utf-8 -*-
"""
俄语课堂实时翻译 · 服务端（跑在寝室 5060 主机上，Windows）

流程：笔记本推来的 PCM 音频 -> silero-VAD 切句 -> faster-whisper 俄语识别
      -> DeepSeek 翻译成中文 -> 回传给笔记本显示

启动示例：
    python server.py --deepseek-key sk-xxxx --subject "宏观经济学"
"""
import argparse
import asyncio
import json
import os
import re
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse, parse_qs

import numpy as np
import wave
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
from openai import OpenAI

SAMPLE_RATE = 16000
WINDOW = 512  # silero-vad 在 16kHz 下固定每次处理 512 个采样点

REC_DIR = "records"   # 课程记录保存目录（main() 里可用 --records-dir 改）
SUBJECT = ""          # 当前课程名（main() 里由 --subject 写入）


# ---------------- 课程记录 ----------------
def open_record(rec_dir: str, subject: str):
    """每建立一个连接就新开一个 Markdown 记录文件，一节课一个文件。"""
    if not rec_dir:
        return None
    os.makedirs(rec_dir, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", subject).strip() or "课程"
    path = os.path.join(rec_dir, f"{time.strftime('%Y%m%d_%H%M')}_{safe}.md")
    f = open(path, "a", encoding="utf-8")
    f.write(f"# {subject or '课堂'} 实时转写记录\n\n"
            f"开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
    f.flush()
    return f


def open_audio(rec_dir: str, subject: str):
    """同步保存整节课的原始录音（16kHz 单声道 WAV），供课后精翻用。"""
    if not rec_dir:
        return None
    os.makedirs(rec_dir, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", subject).strip() or "课程"
    path = os.path.join(rec_dir, f"{time.strftime('%Y%m%d_%H%M')}_{safe}.wav")
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)          # int16
    w.setframerate(SAMPLE_RATE)
    return w


def list_records(rec_dir: str):
    items = []
    if not os.path.isdir(rec_dir):
        return items
    for name in sorted(os.listdir(rec_dir), reverse=True):
        if not name.endswith(".md"):
            continue
        path = os.path.join(rec_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                first = f.readline().lstrip("# ").strip()
            items.append({"name": name, "title": first or name,
                          "size": os.path.getsize(path)})
        except OSError:
            continue
    return items


def read_record(rec_dir: str, name: str):
    # 防目录穿越：只允许纯文件名
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    path = os.path.join(rec_dir, name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------- 翻译 ----------------
def make_translator(api_key: str, base_url: str, model: str, subject: str, glossary: str):
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
    system = (
        f"你是俄语同传译员。以下是俄罗斯大学「{subject}」专业课的课堂口述内容。"
        "把【要翻译的句子】翻译成自然流畅的简体中文，术语要准确；"
        "句子是语音识别出来的碎片，可能有识别错误，请结合前文上下文合理推断原意；"
        "只输出译文本身，不要任何解释、注音或原文。"
    )
    if glossary:
        system += f"\n本课术语对照表（翻译时优先采用）：\n{glossary}"

    def translate(text: str, context: str = "") -> str:
        if context:
            user_msg = (f"【前文上下文，仅供理解，不用翻译】\n{context}\n\n"
                        f"【要翻译的句子】\n{text}")
        else:
            user_msg = f"【要翻译的句子】\n{text}"
        last_err = None
        for attempt in range(3):  # 超时/断线自动重试，间隔递增
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                print(f"[翻译重试 {attempt + 1}/3] {e}", flush=True)
                time.sleep(1.5 * (attempt + 1))
        return f"[翻译失败：{last_err}]"

    return translate


def make_fix_translator(api_key: str, base_url: str, model: str, subject: str, glossary: str):
    """纠错+翻译一体化：先让 LLM 结合上下文/术语表修识别错误，再翻译。
    仿"口语专家"思路——模型对 ASR 错误鲁棒，比"先错着翻"质量好一档。
    返回 (修正后俄语, 中文译文)；失败返回 None，调用方回退普通翻译。"""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
    system = (
        f"你是俄语同传译员。以下是俄罗斯大学「{subject}」专业课的课堂口述内容，"
        "由语音识别转写而成，转写可能有识别错误（听错的词、专业术语、人名）。\n"
        "对【要处理的句子】做两步：\n"
        "1) 结合【前文上下文】和术语表，把转写错误修正为正确的俄语"
        "（只改错词，不改句式，不添加内容）；\n"
        "2) 把修正后的俄语翻译成自然流畅的简体中文。\n"
        "输出严格两行：第一行以 RU: 开头给出修正后的俄语，"
        "第二行以 ZH: 开头给出中文译文。不要输出任何其他内容。"
    )
    if glossary:
        system += f"\n本课术语对照表（修正和翻译都优先采用）：\n{glossary}"

    def fix_translate(text: str, context: str = ""):
        user_msg = ""
        if context:
            user_msg += f"【前文上下文，仅供理解】\n{context}\n\n"
        user_msg += f"【要处理的句子】\n{text}"
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                    max_tokens=768,
                )
                out = resp.choices[0].message.content
                ru_fixed, zh = None, None
                for line in out.splitlines():
                    s = line.strip()
                    if s.upper().startswith("RU:"):
                        ru_fixed = s[3:].strip()
                    elif s.upper().startswith("ZH:"):
                        zh = s[3:].strip()
                if ru_fixed and zh:
                    return ru_fixed, zh
                print(f"[纠错] 输出解析失败，回退普通翻译: {out[:60]}", flush=True)
                return None
            except Exception as e:
                print(f"[纠错重试 {attempt + 1}/3] {e}", flush=True)
                time.sleep(1.5 * (attempt + 1))
        return None

    return fix_translate


# ---------------- 课堂助理（纪要 / 提问提醒 / 随堂问答） ----------------
def make_llm_asker(api_key: str, base_url: str, model: str):
    """通用 LLM 调用，供课堂助理用（同步，放线程里跑）。"""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    def ask(system: str, user: str, max_tokens: int = 2048):
        for attempt in range(2):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.3, max_tokens=max_tokens)
                return r.choices[0].message.content.strip()
            except Exception as e:
                print(f"[助理重试 {attempt + 1}/2] {e}", flush=True)
                time.sleep(1.5 * (attempt + 1))
        return None

    return ask


class Assistant:
    """每个连接一个。维护全文 transcript，周期性生成纪要、判断老师是否提问、回答学生提问。"""
    DIGEST_EVERY_N = 12       # 每满 12 句触发一次
    DIGEST_EVERY_S = 300      # 或距上次digest满 5 分钟

    def __init__(self, ask, subject_getter, send):
        self.ask = ask                    # sync callable(system, user, max_tokens)
        self.subject = subject_getter     # callable -> 当前课程名
        self.send = send                  # async callable(dict)
        self.lines = []                   # [(时间, 俄语, 中文), ...] 全量
        self.last_digest_n = 0
        self.last_digest_t = time.monotonic()
        self.q_sig = ""                   # 已提醒过的问题签名，去重
        self.busy = False

    def feed(self, ru: str, zh: str):
        """每出一条 final 调一次（同步，快速返回）。"""
        self.lines.append((time.strftime("%H:%M:%S"), ru, zh))
        n_new = len(self.lines) - self.last_digest_n
        if (n_new >= self.DIGEST_EVERY_N
                or time.monotonic() - self.last_digest_t >= self.DIGEST_EVERY_S):
            asyncio.ensure_future(self._tick())

    def _fmt(self, lines):
        return "\n".join(f"[{t}] {ru} ⇒ {zh}" for t, ru, zh in lines)

    async def _tick(self):
        if self.busy:
            return
        self.busy = True
        try:
            self.last_digest_n = len(self.lines)
            self.last_digest_t = time.monotonic()
            subj = self.subject() or "大学"
            system = (
                f"你是「{subj}」课堂的学习助理，服务对象是听不懂俄语的中国留学生。\n"
                "输入是课堂实时转写（每行：时间 俄语⇒中文），按顺序给出两块输出：\n"
                "1) 第一行必须是 META: 提问=<无 或 老师刚提出的问题的原文（中文）> | "
                "回答=<如果老师是在提问，给一份简短的中文参考回答；否则留空>\n"
                "2) 然后一行 ---，之后是中文 Markdown 课堂纪要：先一句结论（这节课在讲什么），"
                "再按主题分节（## 小节标题 + 要点列表 - ），术语保留中文，"
                "覆盖截止目前的全部内容（不是增量），不要编造没出现的内容。"
            )
            out = await asyncio.to_thread(self.ask, system, self._fmt(self.lines), 2048)
            if not out:
                return
            meta, _, md = out.partition("\n---")
            md = md.lstrip("\n-").strip()
            q, a = "", ""
            m = re.search(r"META:\s*提问=(.*?)\s*\|\s*回答=(.*)", meta)
            if m:
                q, a = m.group(1).strip().strip("<>"), m.group(2).strip().strip("<>")
            if md:
                await self.send({"kind": "digest", "md": md,
                                 "range": f"{self.lines[0][0]} ~ {self.lines[-1][0]}"})
            if q and q != "无" and q != self.q_sig:
                self.q_sig = q
                await self.send({"kind": "card", "q": q, "a": a})
        finally:
            self.busy = False

    async def answer(self, q: str):
        """回答客户端提问：'没听懂刚才'/'现在要干嘛' 走预设，其他直通。"""
        q = (q or "").strip()
        if not q:
            return
        if not self.lines:
            await self.send({"kind": "answer", "q": q, "a": "还没有收到课堂内容，等老师开始讲课后再问~"})
            return
        subj = self.subject() or "大学"
        if "没听懂" in q or "刚才" in q:
            system = (f"你是「{subj}」课堂的学习助理。学生说没听懂刚才的内容。"
                      "把输入里最近若干句课堂内容用中文大白话解释清楚：先一句话总结老师在说啥，"
                      "再拆开讲要点（术语给中文），必要时举个简单例子。语气像耐心的同学。")
            body = self._fmt(self.lines[-15:])
        elif "干嘛" in q or "作业" in q or "任务" in q or "考试" in q:
            system = (f"你是「{subj}」课堂的学习助理。从整节课内容里梳理老师布置过的"
                      "任务/作业/考试安排/下节课要求：有就逐条列清楚（带时间），"
                      "没有就直说'目前没提到'，不要编造。")
            body = self._fmt(self.lines)
        else:
            system = (f"你是「{subj}」课堂的学习助理。结合课堂内容回答学生的问题，"
                      "先给结论再解释，中文回答，保持简洁。")
            body = self._fmt(self.lines) + f"\n\n【学生的问题】\n{q}"
        a = await asyncio.to_thread(self.ask, system, body, 1024)
        await self.send({"kind": "answer", "q": q,
                         "a": a or "（这题没答上来，再问一次试试）"})


# ---------------- 一次课堂会话 ----------------
class Session:
    def __init__(self, asr, translate, send,
                 min_silence_ms=600, partial_every=1.5, max_seg_s=15,
                 beam_size=3, initial_prompt="", fix_translate=None,
                 final_beam_size=None):
        self.asr = asr                 # WhisperModel
        self.translate = translate     # callable: ru -> zh
        self.fix_translate = fix_translate  # callable: (ru, ctx) -> (ru_fixed, zh) | None
        self.send = send               # async callable(dict)
        self.vad = VADIterator(
            load_silero_vad(onnx=True),
            threshold=0.5,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=150,
            sampling_rate=SAMPLE_RATE,
        )
        self.pending = np.zeros(0, dtype=np.float32)   # 未凑满一窗的残余音频
        self.lookback = deque(maxlen=10)               # 约 0.3s 句首缓冲
        self.in_seg = False
        self.seg = []                  # 当前句子的音频块
        self.seg_samples = 0
        self.partial_every = partial_every
        self.max_seg = int(max_seg_s * SAMPLE_RATE)
        self.last_partial_t = 0.0
        self.last_partial_text = ""
        # 诊断统计
        self.dbg_t = time.monotonic()
        self.dbg_samples = 0
        self.dbg_peak = 0.0
        # 最近两句（俄语/中文）作为翻译上下文，改善碎片句的翻译质量
        self.history = deque(maxlen=2)
        # 识别质量参数：partial 用快档，final 用准档（不赶时间）
        self.beam_size = beam_size
        self.final_beam_size = final_beam_size or beam_size
        self.initial_prompt = initial_prompt

    async def feed(self, pcm_bytes: bytes):
        chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # 每 5 秒打印一次收到的音频量和音量峰值，用于排查收音链路
        self.dbg_samples += len(chunk)
        if len(chunk):
            self.dbg_peak = max(self.dbg_peak, float(np.sqrt(np.mean(chunk ** 2))))
        if time.monotonic() - self.dbg_t >= 5:
            print(f"[音频] 最近5秒收到 {self.dbg_samples / SAMPLE_RATE:.1f}s，"
                  f"音量峰值 {self.dbg_peak:.3f}", flush=True)
            self.dbg_t = time.monotonic()
            self.dbg_samples = 0
            self.dbg_peak = 0.0
        self.pending = np.concatenate([self.pending, chunk])
        while len(self.pending) >= WINDOW:
            window, self.pending = self.pending[:WINDOW], self.pending[WINDOW:]
            await self._handle_window(window)

    async def _handle_window(self, w: np.ndarray):
        event = self.vad(w)
        if self.in_seg:
            self.seg.append(w)
            self.seg_samples += len(w)
        self.lookback.append(w)

        if event:
            if "start" in event and not self.in_seg:
                self.in_seg = True
                self.seg = list(self.lookback)  # 带上一点前文，防止切掉句首
                self.seg_samples = sum(len(x) for x in self.seg)
                self.last_partial_text = ""
                self.last_partial_t = time.monotonic()
                print("[VAD] 检测到语音开始", flush=True)
            if "end" in event and self.in_seg:
                audio = np.concatenate(self.seg)
                self._reset_seg()
                print(f"[VAD] 语音结束，送识别（{len(audio) / SAMPLE_RATE:.1f}s）", flush=True)
                await self._finalize(audio)
                return

        if self.in_seg:
            if self.seg_samples >= self.max_seg:
                # 单句过长（老师语速快/不停顿），强制截断先翻
                audio = np.concatenate(self.seg)
                self._reset_seg()
                self.vad.reset_states()
                print(f"[VAD] 超长强制截断，送识别（{len(audio) / SAMPLE_RATE:.1f}s）", flush=True)
                await self._finalize(audio)
            elif (time.monotonic() - self.last_partial_t >= self.partial_every
                  and self.seg_samples > SAMPLE_RATE):
                self.last_partial_t = time.monotonic()
                await self._partial(np.concatenate(self.seg))

    def _reset_seg(self):
        self.in_seg = False
        self.seg = []
        self.seg_samples = 0

    def _asr_run(self, audio: np.ndarray, beam_size: int = None) -> str:
        segments, _ = self.asr.transcribe(
            audio, language="ru",
            beam_size=beam_size if beam_size is not None else self.beam_size,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt or None,
            # 幻觉过滤：低置信度/疑似幻觉的句子直接丢弃，不再把乱码送去翻译
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        return " ".join(s.text for s in segments).strip()

    async def _partial(self, audio: np.ndarray):
        ru = await asyncio.to_thread(self._asr_run, audio)
        if ru and ru != self.last_partial_text:
            self.last_partial_text = ru
            await self.send({"kind": "partial", "ru": ru})

    async def _finalize(self, audio: np.ndarray):
        if len(audio) < SAMPLE_RATE * 0.4:  # 短于 0.4s 多半是咳嗽/翻书声
            return
        ru = await asyncio.to_thread(self._asr_run, audio, self.final_beam_size)
        if not ru:
            print("[识别] 结果为空（多半是噪音或音量太小）", flush=True)
            return
        ctx = "\n".join(f"{r} / {z}" for r, z in self.history)
        ru_out, zh = ru, None
        if self.fix_translate:
            result = await asyncio.to_thread(self.fix_translate, ru, ctx)
            if result:
                ru_out, zh = result
        if zh is None:
            zh = await asyncio.to_thread(self.translate, ru, ctx)
        if not zh.startswith("[翻译失败"):
            self.history.append((ru_out, zh))
        self.last_partial_text = ""
        print(f"[翻译] {ru_out[:40]} -> {zh[:40]}", flush=True)
        await self.send({"kind": "final", "ru": ru_out, "zh": zh})


# ---------------- 入口 ----------------
async def handle(ws, asr, translate_factory, fix_factory=None, session_opts=None,
                 ask=None):
    # 每个连接独立维护课程名和记录文件；课程名可由客户端随时改
    state = {"subject": SUBJECT, "rec": None}
    translate = translate_factory(state["subject"])
    fix_translate = fix_factory(state["subject"]) if fix_factory else None

    async def send(obj: dict):
        # 每翻完一句先落盘再回传：即使客户端恰在此刻断线，记录也不丢
        if obj.get("kind") == "final":
            if state["rec"] is None:
                state["rec"] = open_record(REC_DIR, state["subject"])
            rec = state["rec"]
            if rec:
                rec.write(f"**{time.strftime('%H:%M:%S')}**  {obj.get('ru', '')}\n\n"
                          f"{obj.get('zh', '')}\n\n---\n\n")
                rec.flush()
            assistant.feed(obj.get("ru", ""), obj.get("zh", ""))
        try:
            await ws.send(json.dumps(obj, ensure_ascii=False))
        except websockets.ConnectionClosed:
            pass  # 客户端已走，记录已落盘，无需处理

    assistant = Assistant(ask, lambda: state["subject"], send)

    session = Session(asr, translate, send, fix_translate=fix_translate,
                      **(session_opts or {}))
    peer = getattr(ws, "remote_address", "?")
    print(f"[连接] 新客户端接入: {peer}", flush=True)
    await send({"kind": "status", "text": "主机已就绪，开始上课吧"})
    wav = None  # 懒创建：收到第一段音频才开文件
    try:
        async for message in ws:
            if isinstance(message, str):
                # 客户端发来的 JSON 配置（目前支持改课程名）
                try:
                    cfg = json.loads(message)
                except ValueError:
                    continue
                if cfg.get("type") == "config":
                    subj = str(cfg.get("subject", "")).strip()
                    if subj and subj != state["subject"]:
                        state["subject"] = subj
                        session.translate = translate_factory(subj)
                        if fix_factory:
                            session.fix_translate = fix_factory(subj)
                        print(f"[配置] 课程名改为: {subj}", flush=True)
                        await send({"kind": "status", "text": f"当前课程：{subj}"})
                elif cfg.get("type") == "ask":
                    asyncio.ensure_future(assistant.answer(str(cfg.get("q", ""))))
                continue
            if isinstance(message, (bytes, bytearray)):
                if wav is None:
                    wav = open_audio(REC_DIR, state["subject"])
                if wav:
                    wav.writeframes(bytes(message))
                try:
                    await session.feed(message)
                except Exception:
                    # 识别出错只打印、不掐断连接，避免客户端反复重连
                    print("[错误] 识别过程异常（连接保持，错误详情如下）:", flush=True)
                    traceback.print_exc()
    except websockets.ConnectionClosed:
        pass
    finally:
        # 断线自动收尾：把说到一半的句子强制送识别+翻译，别浪费已收到的音频
        try:
            if session.in_seg and session.seg_samples >= int(0.4 * SAMPLE_RATE):
                print(f"[断开] 收尾：强制提交未说完的 {session.seg_samples / SAMPLE_RATE:.1f}s 音频",
                      flush=True)
                await session._finalize(np.concatenate(session.seg))
        except Exception:
            print("[错误] 断线收尾失败:", flush=True)
            traceback.print_exc()
        print(f"[断开] 客户端离开: {peer}", flush=True)
        if wav:
            wav.close()
        if state["rec"]:
            state["rec"].write(f"\n结束时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            state["rec"].close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default="large-v3",
                   help="faster-whisper 模型。large-v3 抗噪/远场最好（默认，约 4.5G 显存）；"
                        "large-v3-turbo 更快但嘈杂环境幻觉偏多")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="int8_float16")
    p.add_argument("--beam-size", type=int, default=3,
                   help="识别束搜索宽度，越大越准越慢，实时建议 3")
    p.add_argument("--deepseek-key", default=os.environ.get("TRANSLATE_API_KEY", ""),
                   help="翻译 API Key；不传则从环境变量 TRANSLATE_API_KEY 读取")
    p.add_argument("--translate-base-url", default="https://api.deepseek.com",
                   help="可换成任何 OpenAI 兼容接口（Kimi/通义/本地 vLLM 都行）")
    p.add_argument("--translate-model", default="deepseek-chat")
    p.add_argument("--subject", default="", help="课程名称，喂给翻译模型做上下文")
    p.add_argument("--glossary-file", default="",
                   help="术语对照表 txt 路径（每行：俄语=中文）")
    p.add_argument("--records-dir", default="records",
                   help="课程记录保存目录，默认 ./records；设为空字符串可关闭记录")
    p.add_argument("--http-port", type=int, default=8766,
                   help="HTTP 兼容通道端口（WebSocket 被代理吞掉时的备用方案）")
    args = p.parse_args()

    if not args.deepseek_key:
        p.error("缺少翻译 API Key：用 --deepseek-key 传入，"
                "或先 set TRANSLATE_API_KEY=sk-...")

    global REC_DIR, SUBJECT
    REC_DIR, SUBJECT = args.records_dir, args.subject

    glossary = ""
    if args.glossary_file:
        with open(args.glossary_file, encoding="utf-8") as f:
            glossary = f.read().strip()

    print("[版本] v14 · +课堂助理(自动纪要/提问提醒/随堂问答) · 2026-10")
    print(f"[加载模型] {args.model} ({args.device}/{args.compute_type}) ...")
    asr = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    def translate_factory(subj: str):
        # 每个连接/每次改课程名都独立创建一个翻译器
        return make_translator(args.deepseek_key, args.translate_base_url,
                               args.translate_model, subj or "大学", glossary)
    def fix_factory(subj: str):
        # 纠错+翻译一体化（失败自动回退普通翻译）
        return make_fix_translator(args.deepseek_key, args.translate_base_url,
                                   args.translate_model, subj or "大学", glossary)

    # 识别预热提示：课程名 + 术语表里的俄语术语，显著提升专业词汇识别率
    initial_prompt = ""
    ru_terms = [line.split("=", 1)[0].strip() for line in glossary.splitlines()
                if "=" in line][:50]
    prompt_parts = []
    if args.subject:
        prompt_parts.append(f"Лекция по предмету: {args.subject}.")
    if ru_terms:
        prompt_parts.append("Термины: " + ", ".join(ru_terms) + ".")
    initial_prompt = " ".join(prompt_parts)
    if initial_prompt:
        print(f"[术语预热] 已注入 {len(ru_terms)} 个俄语术语到识别模型")
    # partial 用 --beam-size（默认3）保证速度，final 固定 5 追准确率
    session_opts = {"beam_size": args.beam_size, "initial_prompt": initial_prompt,
                  "final_beam_size": 5}
    asker = make_llm_asker(args.deepseek_key, args.translate_base_url,
                           args.translate_model)

    # 手机网页版客户端：手机浏览器访问 http(s)://主机地址/ 即可使用，
    # 不需要在手机上安装任何东西
    web_html = b""
    web_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone_client.html")
    if os.path.exists(web_path):
        with open(web_path, "rb") as f:
            web_html = f.read()

    def build_response(args, status: int, body: bytes, ctype: str):
        """按 websockets 新旧两版签名构造 HTTP 响应。"""
        if hasattr(args[0], "respond"):           # 新版: 直接返回 Response 对象
            reasons = {200: "OK", 404: "Not Found"}
            return Response(status, reasons.get(status, "OK"),
                            Headers([("Content-Type", ctype),
                                     ("Cache-Control", "no-store")]), body)
        return (status, [("Content-Type", ctype),
                         ("Cache-Control", "no-store")], body)  # 旧版: 元组

    async def process_request(*args):
        # 浏览器普通访问走这里；WebSocket 升级请求必须放行。
        if hasattr(args[0], "respond"):
            path = getattr(args[1], "path", "")   # 新版
            headers = getattr(args[1], "headers", {})
        else:
            path, headers = args[0], args[1]      # 旧版

        # 关键：带 Upgrade: websocket 头的是握手请求，不能拦
        if "websocket" in str(headers.get("Upgrade", "")).lower():
            return None

        if path in ("/", "/index.html"):
            if not web_html:
                return build_response(args, 404, "phone_client.html 未找到".encode(),
                                      "text/plain; charset=utf-8")
            return build_response(args, 200, web_html, "text/html; charset=utf-8")

        if path == "/api/records":                # 历史记录列表
            data = json.dumps(list_records(REC_DIR), ensure_ascii=False).encode()
            return build_response(args, 200, data, "application/json; charset=utf-8")

        if path.startswith("/api/record/"):       # 单条记录全文
            name = unquote(path[len("/api/record/"):])
            content = read_record(REC_DIR, name)
            if content is None:
                return build_response(args, 404, b'{"error": "not found"}',
                                      "application/json")
            data = json.dumps({"name": name, "content": content},
                              ensure_ascii=False).encode()
            return build_response(args, 200, data, "application/json; charset=utf-8")

        # 非 WebSocket 的未知请求（如 favicon.ico）直接 404，不走握手流程
        return build_response(args, 404, b"not found", "text/plain; charset=utf-8")

    # ---------- HTTP 兼容通道（独立端口，纯 GET/POST，给不吃 WebSocket 的代理用） ----------
    # 音频：POST /api/audio?id=xxx（二进制 PCM）  字幕：GET /api/poll?id=xxx（轮询）
    http_clients = {}
    rt = {"loop": None}

    def new_http_client(cid: str):
        c = {"subject": SUBJECT, "outbox": [], "rec": None, "wav": None}

        async def send(obj: dict):
            c["outbox"].append(obj)
            if obj.get("kind") == "final":
                if c["rec"] is None:
                    c["rec"] = open_record(REC_DIR, c["subject"])
                if c["rec"]:
                    c["rec"].write(f"**{time.strftime('%H:%M:%S')}**  {obj.get('ru', '')}\n\n"
                                   f"{obj.get('zh', '')}\n\n---\n\n")
                    c["rec"].flush()
                c["assistant"].feed(obj.get("ru", ""), obj.get("zh", ""))

        c["assistant"] = Assistant(asker, lambda: c["subject"], send)
        c["session"] = Session(asr, translate_factory(c["subject"]), send,
                               fix_translate=fix_factory(c["subject"]),
                               **(session_opts or {}))
        print(f"[连接] HTTP 客户端接入: {cid[:8]}", flush=True)
        return c

    def get_http_client(cid: str):
        c = http_clients.get(cid)
        if c is None:
            c = new_http_client(cid)
            http_clients[cid] = c
        return c

    class ApiHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 关闭访问日志，保持终端干净
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed, qs = urlparse(self.path), parse_qs(urlparse(self.path).query)
            cid = qs.get("id", [""])[0]
            if parsed.path == "/api/poll" and cid:
                c = http_clients.get(cid)
                msgs = [] if c is None else c["outbox"]
                if c is not None:
                    c["outbox"] = []
                self._json({"msgs": msgs})
            elif parsed.path == "/api/config" and cid:
                subj = qs.get("subject", [""])[0].strip()
                q = qs.get("q", [""])[0].strip()
                c = get_http_client(cid)
                if subj and subj != c["subject"]:
                    c["subject"] = subj
                    c["session"].translate = translate_factory(subj)
                    c["session"].fix_translate = fix_factory(subj)
                    print(f"[配置] (HTTP) {cid[:8]} 课程名: {subj}", flush=True)
                if q:
                    asyncio.run_coroutine_threadsafe(c["assistant"].answer(q),
                                                     rt["loop"])
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            parsed, qs = urlparse(self.path), parse_qs(urlparse(self.path).query)
            cid = qs.get("id", [""])[0]
            if parsed.path == "/api/audio" and cid:
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                c = get_http_client(cid)
                if c["wav"] is None:
                    c["wav"] = open_audio(REC_DIR, c["subject"])
                if c["wav"]:
                    c["wav"].writeframes(body)
                if rt["loop"] is not None:
                    asyncio.run_coroutine_threadsafe(c["session"].feed(body), rt["loop"])
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

    async def _serve():
        rt["loop"] = asyncio.get_running_loop()
        httpd = ThreadingHTTPServer((args.host, args.http_port), ApiHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[就绪] HTTP 兼容通道: 端口 {args.http_port}（Tailscale 用 8443 映射它）")
        async with websockets.serve(
            lambda ws: handle(ws, asr, translate_factory, fix_factory, session_opts, asker),
            args.host, args.port, max_size=2 ** 22, ping_interval=20,
            process_request=process_request,
        ):
            print(f"[就绪] ws://{args.host}:{args.port}")
            print(f"[就绪] 手机浏览器打开 http://本机地址:{args.port}/ 即可用手机听课")
            if REC_DIR:
                print(f"[记录] 课程记录保存在 ./{REC_DIR}/ 目录（Markdown 格式）")
            await asyncio.Future()

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
