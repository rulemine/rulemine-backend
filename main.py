from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import base64
import json
import PyPDF2
import io
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Welcome to the Rulemine Chatbot API"}

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

class FilePayload(BaseModel):
    data: str  # base64 encoded file
    mediaType: str
    filename: str

class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    file: Optional[FilePayload] = None
    history: Optional[List[HistoryMessage]] = None

def extract_text_from_pdf(file_data: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2"""
    try:
        pdf_file = io.BytesIO(file_data)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_output = ""
        for page in pdf_reader.pages:
            text_output += page.extract_text() + "\n"
        
        return text_output.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")

SYSTEM_PROMPT = """You are an expert on Indian mining laws and regulations (MMDR Act, MCDR 2017, DGMS circulars, IBM guidelines).
You provide accurate, concise, and well-structured information about Indian mining regulations.
Look closely at what the user is asking — don't give generic or out-of-context responses.
You can greet briefly but stay focused on your area of expertise.
If the user uploads a document, use its content to answer their questions.
Remember the full conversation context and refer back to earlier messages when relevant."""

async def generate_stream(query: str, history: List[dict], extracted_text: Optional[str] = None):
    """Generate streaming response from OpenRouter API with full conversation context"""
    
    # Build the messages array with conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (previous messages)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Build the current user message
    if extracted_text:
        current_msg = f"""User has uploaded a document with the following content:
{extracted_text[:4000]}

User's question: {query}

Please answer based on the document and your knowledge of Indian mining regulations."""
    else:
        current_msg = query
    
    messages.append({"role": "user", "content": current_msg})

    try:
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://rulemine.vercel.app",
                "X-Title": "Rulemine Chatbot",
            },
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=messages,
            stream=True,
            temperature=0.7,
        )

        citations = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content
        
        # Optionally add citations at the end
        if extracted_text:
            citations.append(f"Source: {extracted_text[:100]}...")
        
        if citations:
            yield f"\n\nCITATIONS: {json.dumps(citations)}"
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            yield "⏳ The free AI service is temporarily rate-limited. Please wait 1-2 minutes and try again. (Free tier: ~20 requests/min, 200/day)"
        else:
            yield f"Sorry, something went wrong. Please try again."

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that handles text, file input, and conversation history"""
    
    extracted_text = None
    
    # If file is uploaded, extract text (file is optional)
    if request.file and request.file.data and request.file.data not in ("", "string"):
        try:
            # Check if it's a PDF
            if "pdf" not in request.file.mediaType.lower():
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            # Decode base64 file data (add padding if missing)
            file_data = request.file.data
            file_data += "=" * (-len(file_data) % 4)
            file_bytes = base64.b64decode(file_data)
            extracted_text = extract_text_from_pdf(file_bytes)
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    
    # Build history from request (default to empty)
    history = []
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    return StreamingResponse(
        generate_stream(request.query, history, extracted_text),
        media_type="text/plain"
    )

from fastapi.middleware.cors import CORSMiddleware
allowed_origins = [
    "http://localhost:3000",
    "https://rulemine.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
