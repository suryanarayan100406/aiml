"""
Simple HTTP server for serving the ANI Flow Optimizer frontend.
Models must be loaded via HTTP (not file://) for ONNX Runtime Web.

Usage:
    python serve.py
    
Then open http://localhost:8080 in Chrome.
"""
import http.server
import socketserver
import os
import sys

PORT = 8080
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers and proper MIME types."""
    
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.onnx': 'application/octet-stream',
        '.data': 'application/octet-stream',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
    }
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        # Prevent caching of JS/CSS/HTML during development
        if self.path.endswith(('.js', '.css', '.html', '.md')):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    os.chdir(PROJECT_ROOT)
    
    print("=" * 60)
    print("🧠 ANI Creative Flow Optimizer — Local Server")
    print("=" * 60)
    print(f"   Serving from: {PROJECT_ROOT}")
    print(f"   Frontend:     http://localhost:{PORT}/")
    print(f"   Models dir:   {os.path.join(PROJECT_ROOT, 'models')}")
    print()
    
    # Check for model files
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    expected_models = [
        'desk_distraction_v1.onnx',
        'speech_classifier.onnx', 
        # NLP: replaced 256MB DistilBERT with lightweight keyword classifier (no model file needed)
        'meta_flow_classifier.onnx',
        'screen_classifier.onnx',
    ]
    
    found = 0
    for m in expected_models:
        path = os.path.join(models_dir, m)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"   ✅ {m} ({size:.1f} MB)")
            found += 1
        else:
            print(f"   ❌ {m} — NOT FOUND")
    
    if found < len(expected_models):
        print(f"\n   ⚠️ {len(expected_models) - found} model(s) missing.")
        print(f"   Train models on Colab using scripts in colab/ directory,")
        print(f"   then copy the .onnx files to models/ directory.")
        print(f"   The frontend will run in DEMO mode for missing models.\n")
    else:
        print(f"\n   ✅ All models found! Frontend will run in REAL inference mode.\n")
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"   🌐 Server running at http://localhost:{PORT}/")
        print(f"   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n   Server stopped.")

if __name__ == '__main__':
    main()
