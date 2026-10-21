"""Model-agnostic VLM client over OpenAI-compatible chat/completions with images.

Works with any provider exposing the OpenAI vision API: OpenAI GPT-4o/GPT-5,
DeepSeek-VL (compatible endpoint), Qwen-VL (dashscope compatible-mode / vLLM /
sglang), MiniMax-VL, GLM-4V, local vLLM, etc. Configure via env vars:

  TRACKGUARD_VLM_MODEL    model name, or "mock" for offline pipeline tests
  TRACKGUARD_VLM_BASE_URL e.g. https://api.openai.com/v1 or http://localhost:8000/v1
  TRACKGUARD_VLM_API_KEY  API key
  TRACKGUARD_VLM_PROXY    optional proxy URL; "direct" forces no proxy
"""
import base64
import io
import json
import os
import time
import urllib.request
import urllib.error


class VLMError(Exception):
    pass


def _encode_image(path, max_side=896):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        r = max_side / float(max(im.size))
        im = im.resize((int(im.size[0] * r), int(im.size[1] * r)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


class VLMClient:
    def __init__(self, model=None, base_url=None, api_key=None, proxy=None,
                 timeout=180, retries=3, max_tokens=512):
        self.model = (model or os.environ.get("TRACKGUARD_VLM_MODEL") or "mock").strip()
        self.base_url = (base_url or os.environ.get("TRACKGUARD_VLM_BASE_URL")
                         or "").rstrip("/")
        self.api_key = api_key or os.environ.get("TRACKGUARD_VLM_API_KEY") or ""
        self.proxy = (proxy if proxy is not None
                      else os.environ.get("TRACKGUARD_VLM_PROXY", ""))
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.calls = 0

    def available(self):
        return self.model == "mock" or bool(self.base_url)

    def _opener(self):
        if self.proxy and self.proxy != "env":
            if self.proxy == "direct":
                return urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy}))
        return urllib.request.build_opener()

    def chat(self, prompt, image_paths):
        if self.model == "mock":
            self.calls += 1
            return self._mock(prompt, image_paths)
        if not self.base_url:
            raise VLMError("VLM base_url not configured")
        content = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,"
                                                 + _encode_image(p)}})
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": self.max_tokens}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        opener = self._opener()
        last = None
        for attempt in range(self.retries):
            try:
                with opener.open(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                self.calls += 1
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise VLMError(f"HTTP {e.code}: {e.read()[:200]!r}") from e
            except Exception as e:
                last = e
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise VLMError(str(e)) from last
        raise VLMError(str(last))

    def _mock(self, prompt, image_paths):
        """Offline mode for pipeline/benchmark tests without spending tokens.

        Verdicts come from an optional oracle file next to the evidence dir:
        <out>/<seq>/mock_bad_frames.json = {"visible": [idx...], "infrared": [...]}
        A query (which may include a reference panel plus the queried panel)
        judges BAD when ANY queried frame is listed. Frames without an oracle
        file judge GOOD. Real accuracy claims require a real VLM backend.
        """
        import re as _re
        for p in image_paths:
            m = _re.search(r"^(visible|infrared)_f(\d+)\.png$", os.path.basename(p))
            if not m:
                continue
            mod, idx = m.group(1), int(m.group(2))
            seq_dir = os.path.dirname(os.path.dirname(os.path.abspath(p)))
            oracle = os.path.join(seq_dir, "mock_bad_frames.json")
            if os.path.exists(oracle):
                with open(oracle) as f:
                    bad = json.load(f).get(mod, [])
                if idx in bad:
                    return json.dumps({"verdict": "BAD",
                                       "reason": "mock oracle: known bad frame"})
        return json.dumps({"verdict": "GOOD", "reason": "mock default"})

    def judge(self, prompt, image_paths):
        """Return dict(verdict, reason, raw). verdict in GOOD/BAD/UNCERTAIN."""
        raw = self.chat(prompt, image_paths)
        try:
            m = _re_search_json(raw)
            if m:
                d = json.loads(m)
                v = str(d.get("verdict", "")).upper()
                if v in ("GOOD", "BAD", "UNCERTAIN"):
                    return {"verdict": v, "reason": str(d.get("reason", ""))[:300],
                            "raw": raw}
        except Exception:
            pass
        up = raw.upper()
        if "UNCERTAIN" in up:
            v = "UNCERTAIN"
        elif "BAD" in up:
            v = "BAD"
        elif "GOOD" in up:
            v = "GOOD"
        else:
            v = "UNCERTAIN"
        return {"verdict": v, "reason": raw[:300], "raw": raw}


def _re_search_json(text):
    import re
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else None
