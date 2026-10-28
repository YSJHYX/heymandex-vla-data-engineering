#!/usr/bin/env python3
"""AutoDL 本机邮件桥；只用标准库，默认不发送测试邮件。"""

import json
import math
import os
import smtplib
import ssl
import subprocess
import sys
import threading
import time
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORIGIN = "https://www.autodl.com"
MAX_BODY = 16384


def smtp_password(email):
    """只从环境或 macOS 钥匙串读取，不写配置、不打印授权码。"""
    password = os.environ.get("AUTODL_SMTP_PASSWORD")
    if password:
        return password
    if sys.platform == "darwin":
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "dsh-smtp", "-a", email, "-w"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise RuntimeError("缺少 SMTP 授权码：请设置 AUTODL_SMTP_PASSWORD 或 macOS 钥匙串 dsh-smtp")


def send_notification(payload):
    if os.environ.get("AUTODL_MAIL_DRY") == "1":
        print("[邮件桥] 演练，不发邮件：" + json.dumps(payload, ensure_ascii=False), flush=True)
        return
    sender = os.environ.get("AUTODL_MAIL_FROM", "").strip()
    recipient = os.environ.get("AUTODL_MAIL_TO", sender).strip()
    if not sender or not recipient or any(c in sender + recipient for c in "\r\n"):
        raise RuntimeError("请设置有效的 AUTODL_MAIL_FROM 和 AUTODL_MAIL_TO")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = "AutoDL 创建流程已触发，请核实实例"
    message.set_content(
        "脚本已触发创建流程并锁定。这不代表服务端已创建成功，请核实实例列表及订单。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    password = smtp_password(sender)
    with smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=20, context=ssl.create_default_context()) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)
    print("[邮件桥] 邮件已交给 SMTP 服务器", flush=True)


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON 必须是对象")
    # 收件人固定在本机环境；页面不能指定收件地址或 SMTP 参数。
    allowed = {"event", "host", "model", "idle", "price", "message"}
    if not payload or set(payload) - allowed:
        raise ValueError("JSON 字段不符合邮件通知约定")
    for key, value in payload.items():
        if key in {"idle", "price"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("空闲数、价格必须是非负数")
        elif not isinstance(value, str) or len(value) > 2000:
            raise ValueError("通知文字过长或类型错误")
    if not payload.get("message"):
        raise ValueError("缺少 message")
    return payload


class MailServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, sender=send_notification):
        super().__init__(address, handler)
        self.sender = sender
        self.guard = threading.Lock()
        self.recent = {}
        self.last_attempt = 0.0


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoDLLocalMail/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def respond(self, status, message):
        content = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        # no-cors 的页面请求读不到此响应；不开放任意来源 CORS。
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != "/send":
            self.respond(404, "只提供 /send")
            return
        if self.headers.get_all("Origin") != [ORIGIN]:
            self.respond(403, "来源不允许")
            return
        if self.headers.get("Transfer-Encoding"):
            self.respond(400, "不接受分块请求")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.respond(413, "通知为空或过长")
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("通知不完整")
            payload = validate_payload(json.loads(raw.decode("utf-8"),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("非法数值"))))
        except (ValueError, UnicodeError, TimeoutError):
            self.respond(400, "请求体必须是符合约定的 JSON 对象")
            return
        key = (payload.get("event"), payload.get("host"), payload.get("model"))
        now = time.monotonic()
        with self.server.guard:
            self.server.recent = {k: at for k, at in self.server.recent.items() if now - at < 300}
            if key in self.server.recent:
                self.respond(200, "五分钟内重复通知，已忽略")
                return
            if now - self.server.last_attempt < 5:
                self.respond(429, "邮件请求过于频繁")
                return
            # SMTP 超时也不自动重发，以免产生重复邮件。
            self.server.last_attempt = now
            self.server.recent[key] = now
        try:
            self.server.sender(payload)
        except Exception as error:
            # 不把异常正文打印出来，避免供应商错误响应回显凭据。
            print(f"[邮件桥] 发送失败（{type(error).__name__}），请检查本机邮件配置", flush=True)
            self.respond(502, "邮件未确认发送，请检查本机配置")
            return
        self.respond(200, "演练完成" if os.environ.get("AUTODL_MAIL_DRY") == "1" else "邮件已提交")

    def do_OPTIONS(self):
        self.respond(405, "仅支持无需预检的 POST /send")

    def do_GET(self):
        self.respond(405, "仅支持 POST /send")

    def log_message(self, format, *args):
        # 不打印请求头和正文。
        print("[邮件桥] " + format % args, flush=True)


def main():
    mode = "演练，不发邮件" if os.environ.get("AUTODL_MAIL_DRY") == "1" else "SMTP 通知"
    server = MailServer(("127.0.0.1", 8765), Handler)
    print(f"[邮件桥] 监听 http://127.0.0.1:8765/send；{mode}；只接受 {ORIGIN}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
