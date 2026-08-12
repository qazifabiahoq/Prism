import json
import os
import re
import traceback
from http.server import BaseHTTPRequestHandler

EXTRACTION_PROMPT = """You are reading a photo of a receipt, bank statement, or transaction screenshot.

Extract every individual transaction you can find. Return ONLY a JSON array (no markdown, no commentary), where each item has exactly these fields:
- "date": the transaction date as YYYY-MM-DD if visible, otherwise null
- "description": the merchant or line-item name, short and plain text
- "amount": the transaction amount as a positive number (no currency symbols)

If you cannot find any transactions, return an empty array: []
"""


def extract_json_array(text):
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw_body or b"{}")
            image_b64 = (payload.get("image") or "").strip()
            mime_type = payload.get("mimeType") or "image/jpeg"

            if not image_b64:
                self._send_json(400, {"ok": False, "error": "No image was provided."})
                return

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                self._send_json(200, {
                    "ok": False,
                    "error": "Photo scanning requires the AI advisor to be configured for this deployment. Please use CSV upload instead.",
                })
                return

            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            raw = completion.choices[0].message.content
            parsed = extract_json_array(raw)

            if not parsed:
                self._send_json(200, {
                    "ok": False,
                    "error": "Could not find any transactions in that photo. Please ensure it's a clear, well-lit image, or use CSV upload.",
                })
                return

            rows = []
            for item in parsed:
                try:
                    amount = float(item.get("amount"))
                except (TypeError, ValueError):
                    continue
                if amount <= 0:
                    continue
                rows.append({
                    "Date": item.get("date") or None,
                    "Amount": round(amount, 2),
                    "Description": str(item.get("description") or "Unlabeled transaction")[:80],
                })

            if not rows:
                self._send_json(200, {
                    "ok": False,
                    "error": "Could not find any valid transactions in that photo. Please ensure it's a clear, well-lit image, or use CSV upload.",
                })
                return

            self._send_json(200, {"ok": True, "rows": rows})
        except Exception:  # noqa: BLE001
            print("receipt.py error:", traceback.format_exc())
            self._send_json(200, {"ok": False, "error": "Photo scanning is temporarily unavailable. Please use CSV upload."})
