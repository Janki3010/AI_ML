from fastapi import FastAPI, Header, HTTPException
import ollama
import os
from dotenv import load_dotenv
from fastapi.params import Depends

load_dotenv()
API_KEYS_CREDITS = {os.getenv("API_KEY"): 5}

app = FastAPI()

def verify_api_key(x_api_key: str = Header(None)):
    credits = API_KEYS_CREDITS.get(x_api_key, 0)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API key, or not")
    return x_api_key

@app.post("/generate")
def generate(prompt: str, x_api_key: str = Depends(verify_api_key)):
    API_KEYS_CREDITS[x_api_key] -= 1
    print(API_KEYS_CREDITS[x_api_key])
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return {"response": response["message"]["content"]}
