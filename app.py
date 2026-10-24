"""Steady Inbox: local semantic grouping with source-preserving digests."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SAMPLE = [
    'Please review the pull request before Friday.',
    'Could you review the PR before Friday?',
    'Remember to buy milk on the way home.',
    'Pick up some milk on your way back home.',
    'The design meeting has moved to 3 PM.',
    'Our design meeting is now at 3 PM.',
    'Back up the database before the migration.',
    'Make a database backup before migrating.',
    'The new office Wi-Fi password is on the noticeboard.',
]


def validate_request(data: dict) -> tuple[list[str], float]:
    """Bound work and reject incomplete input without changing the notes."""
    if not isinstance(data, dict) or not isinstance(data.get('text'), str):
        raise ValueError('Provide notes as text, one message per line.')
    notes = [line.strip() for line in data['text'].splitlines() if line.strip()]
    if not notes:
        raise ValueError('Add at least one non-empty message.')
    if len(notes) > 100:
        raise ValueError('Use at most 100 messages per digest.')
    if any(len(note) > 1500 for note in notes):
        raise ValueError('Keep each message under 1,500 characters; split longer notes first.')
    threshold = data.get('threshold', .76)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError('Similarity must be a number from 0.50 to 0.95.')
    if not math.isfinite(threshold) or not .5 <= threshold <= .95:
        raise ValueError('Similarity must be a number from 0.50 to 0.95.')
    return notes, float(threshold)


def group_vectors(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy complete-link groups: every pair must clear the threshold."""
    similarities = np.clip(vectors @ vectors.T, -1, 1)
    groups: list[list[int]] = []
    for index in range(len(vectors)):
        candidates = [
            (float(similarities[index, group].mean()), group_index)
            for group_index, group in enumerate(groups)
            if float(similarities[index, group].min()) + 1e-6 >= threshold
        ]
        if candidates:
            groups[max(candidates)[1]].append(index)
        else:
            groups.append([index])
    return groups


class Engine:
    """Load only local model files; notes are never sent to a model service."""
    def __init__(self, directory: Path = ROOT / 'models') -> None:
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / 'model.onnx'), sess_options=options,
            providers=['CPUExecutionProvider'],
        )
        self.lock = threading.Lock()

    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        vectors, truncated = [], []
        with self.lock:
            for start in range(0, len(texts), 16):
                batch = self.tokenizer.encode_batch(texts[start:start + 16])
                ids = np.asarray([item.ids for item in batch], dtype=np.int64)
                masks = np.asarray([item.attention_mask for item in batch], dtype=np.int64)
                feeds = {'input_ids': ids, 'attention_mask': masks,
                         'token_type_ids': np.asarray([item.type_ids for item in batch], dtype=np.int64)}
                hidden = self.session.run(None, feeds)[0]
                expanded = masks[..., None].astype(np.float32)
                pooled = (hidden * expanded).sum(axis=1) / np.maximum(expanded.sum(axis=1), 1e-9)
                pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
                vectors.extend(pooled)
                truncated.extend(bool(item.overflowing) for item in batch)
        return np.asarray(vectors), truncated

    def digest(self, data: dict) -> dict:
        notes, threshold = validate_request(data)
        began = time.perf_counter()
        vectors, truncated = self.embed(notes)
        clusters = group_vectors(vectors, threshold)
        groups = []
        for indices in clusters:
            inside = vectors[indices] @ vectors[indices].T
            representative = indices[int(np.argmax(inside.mean(axis=1)))]
            groups.append({
                'representative': notes[representative],
                'source_ids': [index + 1 for index in indices],
                'messages': [notes[index] for index in indices],
                'min_similarity': round(float(inside.min()), 4),
                'truncated_ids': [index + 1 for index in indices if truncated[index]],
            })
        return {
            'groups': groups, 'message_count': len(notes), 'group_count': len(groups),
            'repeated_count': len(notes) - len(groups), 'threshold': threshold,
            'inference_ms': round((time.perf_counter() - began) * 1000),
            'model': MODEL_ID, 'revision': MODEL_REVISION,
            'truncated_count': sum(truncated),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = 'SteadyInbox/1.0'

    def log_message(self, *_: object) -> None:
        """Do not log note text, query strings, or client identifiers."""

    def send_data(self, status: int, content: bytes, mime: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def json_response(self, status: int, data: dict) -> None:
        self.send_data(status, json.dumps(data).encode(), 'application/json; charset=utf-8')

    def valid_host(self) -> bool:
        return self.headers.get('Host') in {
            f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'
        }

    def do_GET(self) -> None:
        if not self.valid_host():
            self.json_response(403, {'error': 'Open Steady Inbox using its localhost address.'})
            return
        if self.path == '/api/status':
            self.json_response(200, {'ready': True, 'model': MODEL_ID, 'sample': SAMPLE})
            return
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.path not in assets:
            self.json_response(404, {'error': 'Not found'})
            return
        name, mime = assets[self.path]
        self.send_data(200, (ROOT / 'static' / name).read_bytes(), mime)

    def do_POST(self) -> None:
        origin = self.headers.get('Origin')
        allowed_origins = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if not self.valid_host() or origin not in allowed_origins:
            self.json_response(403, {'error': 'Only requests from this local app are accepted.'})
            return
        if self.path != '/api/digest':
            self.json_response(404, {'error': 'Not found'})
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            self.json_response(415, {'error': 'Send JSON.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '-1'))
            if length < 0 or length > 160_000:
                raise ValueError('Request is too large or has no length.')
            data = json.loads(self.rfile.read(length))
            self.json_response(200, self.server.engine.digest(data))
        except (ValueError, UnicodeError) as exc:
            self.json_response(400, {'error': str(exc)})
        except Exception:
            self.json_response(500, {'error': 'Local inference failed. Check the model files and restart.'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8797)
    args = parser.parse_args()
    print('Loading the local MiniLM model...', flush=True)
    engine = Engine()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.engine = engine
    print(f'Steady Inbox is ready at http://127.0.0.1:{args.port}', flush=True)
    print('Notes are processed in memory; the app does not save or transmit them externally.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
