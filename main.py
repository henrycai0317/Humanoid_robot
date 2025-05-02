from openai import OpenAI
from fastapi import FastAPI, Form, Request, WebSocket
from typing import Annotated
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
import os
from dotenv import load_dotenv
import asyncio
import ultravox_client as uv  # 導入 Ultravox 客戶端


load_dotenv()

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_SECRET_KEY")
)
ultravox_api_key = os.getenv("ULTRAVOX_API_KEY")  # 添加這行以獲取 API 密鑰

app = FastAPI()
templates = Jinja2Templates(directory="templates")

chat_responses = []
chat_log = [{'role':'system',
             'content':'You tell jokes.'
             }]

# 創建一個管理 Ultravox 通話的類
class UltravoxCallManager:
    def __init__(self):
        self.active_sessions = {}
    
    async def create_call(self, system_prompt="你是一個友善且有幫助的 AI 助手，使用繁體中文回答問題。"):
        import requests  # 添加 requests 導入
        # 使用 requests 建立通話
        # 修正 payload 格式
        payload = {
            "model": "fixie-ai/ultravox-70B",
            "systemPrompt": system_prompt,
            "medium": {
                "webRtc": {}
            },
            "voice": "Mark"  # 使用字串而非物件，選擇一個預設聲音如 "Mark"
        }
            
        response = requests.post(
            "https://api.ultravox.ai/api/calls",
            headers={
                "X-API-Key": ultravox_api_key,
                "Content-Type": "application/json"
            },
            json=payload
        )
        
    # 直接解析回應並只返回所需的資訊
        result = response.json()
        if 'callId' in result and 'joinUrl' in result:
            call_id = result["callId"]
            join_url = result["joinUrl"]
            
            # 建立 Ultravox 會話
            session = uv.UltravoxSession()
            self.active_sessions[call_id] = session
            
            # 只返回所需的資訊
            return {"joinUrl": join_url, "callId": call_id}
        else:
            raise Exception(f"API 回應中未找到必要資訊: {response.text}")
    
    async def end_call(self, call_id):
        if call_id in self.active_sessions:
            session = self.active_sessions[call_id]
            await session.leave_call()
            del self.active_sessions[call_id]
            
            # 通知 Ultravox API 結束通話
            response = requests.delete(
                f"https://api.ultravox.ai/api/calls/{call_id}",
                headers={
                    "X-API-Key": ultravox_api_key
                }
            )
            
            return True
        return False

# 初始化 Ultravox 通話管理器
ultravox_manager = UltravoxCallManager()

# 文字聊天頁面路由 (只定義一次)
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})

# 添加語音對話頁面路由
@app.get("/voice", response_class=HTMLResponse)
async def voice_page(request: Request):
    return templates.TemplateResponse("voice.html", {"request": request})

# 添加創建語音通話的路由
@app.post("/create-voice-call")
async def create_voice_call(request: Request):
    try:
        result = await ultravox_manager.create_call()
        # 只返回必要的信息
        return {
            "joinUrl": result["joinUrl"], 
            "callId": result["callId"]
        }
    except Exception as e:
        return {"error": str(e)}

# 添加結束語音通話的路由
@app.post("/end-voice-call/{call_id}")
async def end_voice_call(call_id: str):
    try:
        success = await ultravox_manager.end_call(call_id)
        if success:
            return {"success": True}
        else:
            return {"error": "Call not found or already ended"}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        user_input = await websocket.receive_text()
        chat_log.append({'role':'user', 'content': user_input})
        chat_responses.append(user_input)

        try:
            response = openai.chat.completions.create(
                model='gpt-4',
                messages=chat_log,
                temperature=0.6,
                stream=True
            )

            ai_response = ''

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    ai_response += chunk.choices[0].delta.content
                    await websocket.send_text(chunk.choices[0].delta.content)
            chat_responses.append(ai_response)

        except Exception as e:
            await websocket.send_text(f'Error: {str(e)}')
            break

@app.post("/", response_class=HTMLResponse)
async def chat_form(request: Request, user_input: Annotated[str, Form()]):
    chat_log.append({'role': 'user', 'content': user_input})
    chat_responses.append(user_input)

    responses = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=chat_log,
        temperature=.6
    )

    bot_response = responses.choices[0].message.content
    chat_log.append({'role':'assistant', 'content': bot_response})
    chat_responses.append(bot_response)

    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})

@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request):
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/image", response_class=HTMLResponse)
async def create_image(request: Request, user_input: Annotated[str, Form()]):
    response = openai.images.generate(
        prompt=user_input,
        n = 1,
        size="256x256"
    )

    image_url = response.data[0].url
    return templates.TemplateResponse("image.html", {"request": request, "image_url": image_url})