"""
Fase 6.2: Server-Sent Events (SSE) untuk Streaming di Web App
==============================================================
Pelajaran sebelumnya (01_streaming_response.py) kita sudah bisa streaming
response dari LLM + tool calls di script Python biasa (print token by token).

Tapi di dunia nyata, chatbot AI itu jalan di WEB -- browser kirim request,
backend harus nge-stream response REAL-TIME ke frontend tanpa bikin frontend
nunggu HTTP response selesai.

Ada 3 cara umum streaming ke browser:
    1. Server-Sent Events (SSE) -- HTTP connection yg tetap buka, server push
       data bertahap via `data: ...` format. SIMPLE, one-way (server->client).
    2. WebSocket -- full-duplex, tapi overkill untuk streaming LLM (kita gak
       butuh client kirim banyak message saat LLM lagi generate).
    3. HTTP chunked transfer encoding -- low-level, gak ada built-in event
       parsing di browser.

**SSE adalah pilihan standar industri untuk streaming LLM response.**
OpenAI API official, Anthropic Claude, semua pakai SSE format.

Task di file ini:
    a. Buat FastAPI server sederhana dengan endpoint /chat/stream yang:
       - Terima user message (POST JSON)
       - Stream LLM response via SSE (mimetype text/event-stream)
       - Kirim setiap token sebagai event SSE (format: `data: {...}\\n\\n`)
    b. Buat HTML client sederhana yang:
       - Kirim request via fetch() ke /chat/stream
       - Baca SSE stream pakai EventSource atau manual parsing
       - Tampilkan token by token di halaman (simulasi ChatGPT UX)
    c. Bonus: kirim metadata (token count, latency) di event terakhir

Catatan: FastAPI punya StreamingResponse built-in yang cocok banget buat SSE.
Kita cukup yield string berformat `data: {json}\\n\\n` dari async generator.

Dependencies: fastapi, uvicorn, openai, python-dotenv
"""

import json
import os
import time
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from openai import OpenAI

load_dotenv()

BASE_URL = os.environ["ROUTER_BASE_URL"]
API_KEY = os.environ["ROUTER_API_KEY"]
MODEL = os.environ["ROUTER_MODEL"]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

app = FastAPI(title="SSE Streaming Demo")


async def stream_llm_response(user_message: str) -> AsyncGenerator[str, None]:
    """
    Generator yang yield SSE-formatted events. Setiap event:
        data: {"type": "token", "content": "..."}\n\n
    atau
        data: {"type": "done", "metadata": {...}}\n\n

    FastAPI StreamingResponse akan kirim tiap yield langsung ke client tanpa
    buffer (kalau client support SSE).
    """
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = ""

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            token_count += 1
            full_text += delta.content

            # Kirim token sebagai SSE event
            event_data = {
                "type": "token",
                "content": delta.content,
            }
            yield f"data: {json.dumps(event_data)}\n\n"

    end = time.perf_counter()

    # Event terakhir: metadata (TTFT, total time, token count)
    metadata = {
        "type": "done",
        "metadata": {
            "ttft": round(first_token_time - start, 3) if first_token_time else None,
            "total_time": round(end - start, 3),
            "token_count": token_count,
            "total_chars": len(full_text),
        },
    }
    yield f"data: {json.dumps(metadata)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Endpoint streaming. Body JSON: {"message": "..."}
    Response: text/event-stream (SSE format)
    """
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message:
        return {"error": "Message is required"}

    return StreamingResponse(
        stream_llm_response(user_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx: disable buffering
        },
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Simple HTML client untuk test SSE streaming (buka browser ke http://localhost:8000)
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SSE Streaming Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        h1 { color: #60a5fa; }
        #input-area {
            margin: 20px 0;
        }
        #user-input {
            width: 100%;
            padding: 12px;
            font-size: 14px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #2a2a2a;
            color: #e0e0e0;
            box-sizing: border-box;
        }
        button {
            margin-top: 10px;
            padding: 10px 20px;
            font-size: 14px;
            background: #60a5fa;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        button:hover { background: #3b82f6; }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        #output {
            margin-top: 20px;
            padding: 20px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 6px;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .metadata {
            margin-top: 20px;
            padding: 10px;
            background: #1e3a5f;
            border-radius: 4px;
            font-size: 12px;
            color: #9ca3af;
        }
        .cursor {
            display: inline-block;
            width: 2px;
            height: 1em;
            background: #60a5fa;
            animation: blink 1s infinite;
            margin-left: 2px;
        }
        @keyframes blink {
            0%, 49% { opacity: 1; }
            50%, 100% { opacity: 0; }
        }
    </style>
</head>
<body>
    <h1>🚀 SSE Streaming Demo (Fase 6.2)</h1>
    <p>Server-Sent Events untuk streaming LLM response ke browser real-time.</p>
    
    <div id="input-area">
        <textarea id="user-input" rows="3" placeholder="Ketik pesan Anda...">Jelaskan dalam 3 kalimat: kenapa SSE lebih cocok untuk streaming LLM dibanding WebSocket?</textarea>
        <br>
        <button id="send-btn" onclick="sendMessage()">Kirim & Stream</button>
    </div>
    
    <div id="output"></div>
    <div id="metadata" class="metadata"></div>
    
    <script>
        const outputDiv = document.getElementById('output');
        const metadataDiv = document.getElementById('metadata');
        const sendBtn = document.getElementById('send-btn');
        const inputArea = document.getElementById('user-input');
        
        async function sendMessage() {
            const message = inputArea.value.trim();
            if (!message) return;
            
            // Reset UI
            outputDiv.innerHTML = '<span class="cursor"></span>';
            metadataDiv.innerHTML = '';
            sendBtn.disabled = true;
            
            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                if (!response.ok) {
                    throw new Error('Request failed');
                }
                
                // Parse SSE stream manual (EventSource gak support POST)
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let fullText = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    
                    // SSE format: "data: {...}\\n\\n"
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop(); // simpan incomplete chunk
                    
                    for (const line of lines) {
                        if (!line.trim() || !line.startsWith('data: ')) continue;
                        
                        const jsonStr = line.replace('data: ', '').trim();
                        try {
                            const event = JSON.parse(jsonStr);
                            
                            if (event.type === 'token') {
                                fullText += event.content;
                                outputDiv.innerHTML = fullText + '<span class="cursor"></span>';
                            } else if (event.type === 'done') {
                                outputDiv.innerHTML = fullText; // remove cursor
                                const meta = event.metadata;
                                metadataDiv.innerHTML = `
                                    ✅ <b>Done</b> | 
                                    TTFT: <b>${meta.ttft}s</b> | 
                                    Total: <b>${meta.total_time}s</b> | 
                                    Tokens: <b>${meta.token_count}</b> | 
                                    Chars: <b>${meta.total_chars}</b>
                                `;
                            }
                        } catch (e) {
                            console.error('Parse error:', e, jsonStr);
                        }
                    }
                }
            } catch (error) {
                outputDiv.innerHTML = '❌ Error: ' + error.message;
            } finally {
                sendBtn.disabled = false;
            }
        }
        
        // Enter to send (Shift+Enter for newline)
        inputArea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("SSE Streaming Server Starting...")
    print("=" * 70)
    print("\n🌐 Buka browser: http://localhost:8000")
    print("   Ketik pesan, klik 'Kirim & Stream', lihat token muncul real-time.\n")
    print("📌 Cara kerja:")
    print("   1. Browser kirim POST ke /chat/stream dengan JSON {message: '...'}")
    print("   2. Server buka koneksi SSE (text/event-stream)")
    print("   3. Tiap token dari LLM langsung di-push sebagai SSE event")
    print("   4. Browser parsing SSE stream, tampilkan token by token")
    print("   5. Event terakhir 'done' kirim metadata (TTFT, total time)\n")
    print("🔑 Kunci: StreamingResponse + async generator = SSE native di FastAPI")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
