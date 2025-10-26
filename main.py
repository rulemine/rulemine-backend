from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
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

class ChatRequest(BaseModel):
    query: str
    file: Optional[FilePayload] = None

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

async def generate_stream(query: str, extracted_text: Optional[str] = None):
    """Generate streaming response from OpenRouter API"""
    
    if extracted_text:
        prompt = f"""You are a helpful assistant for Indian mining law compliance (MMDR, MCDR, DGMS).
        
User has uploaded a document with the following content:
{extracted_text[:4000]}  # Limit context to avoid token overflow

User's question: {query}

Please answer based on the document and your knowledge of Indian mining regulations."""
    else:
        prompt = f"""You are a helpful assistant for Indian mining law compliance (MMDR, MCDR, DGMS).

User's question: {query}

Please provide accurate information about Indian mining regulations in a well concised structured details."""

    try:
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000"),
                "X-Title": "Rulemine Chatbot",
            },
            model="meta-llama/llama-4-maverick:free",
            messages=[
                {"role": "system", "content": "You are an expert on Indian mining laws and regulations."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.7,
        )

        # Stream the response
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
        yield f"Error: {str(e)}"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that handles text and file input"""
    
    extracted_text = None
    
    # If file is uploaded, extract text
    if request.file:
        try:
            # Decode base64 file data
            file_bytes = base64.b64decode(request.file.data)
            
            # Extract text based on file type
            if request.file.mediaType == "application/pdf":
                extracted_text = extract_text_from_pdf(file_bytes)
            else:
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    
    # Return streaming response
    return StreamingResponse(
        generate_stream(request.query, extracted_text),
        media_type="text/plain"
    )

from fastapi.middleware.cors import CORSMiddleware
allowed_origins = ["http://localhost:3000"]
production_url = os.getenv("SITE_URL")
if production_url and production_url not in allowed_origins:
    allowed_origins.append(production_url)

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
