import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from simple_model import SimpleNeuralNetwork


TRAINING_SAMPLES = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

model = SimpleNeuralNetwork()
loss_history = model.train(TRAINING_SAMPLES)
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            return self.serve_file("index.html", "text/html; charset=utf-8")
        if parsed.path.startswith("/static/"):
            filename = os.path.basename(parsed.path)
            content_type = (
                "application/javascript; charset=utf-8"
                if filename.endswith(".js")
                else "text/css; charset=utf-8"
            )
            return self.serve_file(os.path.join("static", filename), content_type)
        if parsed.path == "/api/model":
            return self.send_json(
                200,
                {
                    "graph": model.graph(),
                    "lossHistory": loss_history,
                    "trainingSamples": [
                        {"x": sample[0], "target": sample[1]}
                        for sample in TRAINING_SAMPLES
                    ],
                },
            )
        if parsed.path == "/api/predict":
            query = parse_qs(parsed.query)
            try:
                x = float(query["x"][0])
                y = float(query["y"][0])
            except (KeyError, ValueError):
                return self.send_json(400, {"error": "Informe os parâmetros x e y."})
            return self.send_json(
                200,
                {
                    "probability": round(model.predict([x, y]), 6),
                    "class": 1 if model.predict([x, y]) >= 0.5 else 0,
                },
            )
        if parsed.path == "/api/grid":
            grid = []
            steps = 21
            for row in range(steps):
                y = row / (steps - 1)
                for column in range(steps):
                    x = column / (steps - 1)
                    grid.append(
                        {
                            "x": x,
                            "y": y,
                            "probability": round(model.predict([x, y]), 5),
                        }
                    )
            return self.send_json(200, {"grid": grid})

        return self.send_json(404, {"error": "Endpoint não encontrado"})

    def serve_file(self, relative_path, content_type):
        path = os.path.join(ROOT, relative_path)
        try:
            with open(path, "rb") as file:
                body = file.read()
        except FileNotFoundError:
            return self.send_json(404, {"error": "Arquivo não encontrado"})
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Servidor disponível em http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    server.serve_forever()
