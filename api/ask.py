import json
import os
import traceback
from http.server import BaseHTTPRequestHandler


def build_prompt(question, context):
    return f"""User's financial data summary:
- Total transactions: {context.get('transactionCount', 'unknown')}
- Average spending: ${context.get('averageAmount', 0):.2f}
- Wellness score: {context.get('wellnessScore', 0)}/100
- Unusual transactions: {context.get('anomalyCount', 0)}
- Top spending category: {context.get('topCategory', 'Unknown')}

User question: {question}

Provide helpful, actionable financial advice in a friendly, professional tone. Be specific and data-driven. Keep it under 150 words.
"""


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
            question = (payload.get("question") or "").strip()
            context = payload.get("context") or {}

            if not question:
                self._send_json(400, {"ok": False, "error": "Ask a question first."})
                return

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                self._send_json(200, {
                    "ok": True,
                    "answer": "The AI advisor is currently being configured for this deployment. In the meantime, all of your forecasting, fraud detection, and dashboard analysis is fully live.",
                })
                return

            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": build_prompt(question, context)}],
                max_tokens=500,
                temperature=0.7,
            )
            answer = completion.choices[0].message.content
            self._send_json(200, {"ok": True, "answer": answer})
        except Exception:  # noqa: BLE001
            print("ask.py error:", traceback.format_exc())
            self._send_json(200, {"ok": False, "error": "Assistant temporarily unavailable. Please try again later."})
