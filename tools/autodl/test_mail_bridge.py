"""只在本机演练 HTTP，不连接 SMTP。"""
import http.client
import json
import threading
import unittest

from mail_bridge import Handler, MailServer, validate_payload


class MailBridgeTest(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.server = MailServer(("127.0.0.1", 0), Handler, sender=self.sent.append)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, body, origin="https://www.autodl.com", path="/send"):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        conn.request("POST", path, body=body, headers={"Origin": origin, "Content-Type": "text/plain"})
        response = conn.getresponse()
        status = response.status
        response.read()
        conn.close()
        return status

    def test_origin_json_and_dedup(self):
        body = json.dumps({"host": "test-host", "message": "模拟通知"})
        self.assertEqual(self.post(body), 200)
        self.assertEqual(self.post(body), 200)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.post(body, origin="https://example.com"), 403)
        self.assertEqual(self.post("not-json"), 400)
        self.assertEqual(self.post("[]"), 400)
        self.assertEqual(self.post(body, path="/other"), 404)

    def test_reject_recipient_override_and_bad_values(self):
        for payload in [{"message": "test", "to": "other@example.com"},
                        {"message": "test", "idle": -1}, {"message": "test", "price": float("inf")}]:
            with self.assertRaises(ValueError):
                validate_payload(payload)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
