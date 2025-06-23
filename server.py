from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(filename='assistant.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI and rate limiter
app = FastAPI(title="Pro AI Assistant API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache (TTL: 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

# Groq API configuration
GROQ_API_KEY = "YOUR GROQ API KEY"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class Query(BaseModel):
    query: str
    model: str = "llama3-8b-8192"

# Single API endpoint: /query
@app.post("/query")
@limiter.limit("10/minute")  # 10 requests per minute
async def get_response(query: Query, request: Request) -> Dict[str, str]:
    """Handle queries using specified Groq model, ensuring structured responses."""
    try:
        # Validate model
        valid_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "gemma2-9b-it",
            "llama-3.3-70b-versatile"
        ]
    
        if query.model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

        # Check cache
        cache_key = f"{query.model}:{query.query.lower()}"
        if cache_key in cache:
            logging.info(f"Cache hit for query: {query.query}")
            return {"response": cache[cache_key]}

        # Craft prompt for structured response
        prompt = f"""
        Provide a detailed, well-structured response to the following query in markdown format. Use headings, lists, and code blocks where appropriate. Ensure clarity and professionalism.

        Query: {query.query}
        """

        # Call Groq API
        logging.info(f"Attempting to call Groq API at: {GROQ_API_URL} with model: {query.model}")
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": query.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": False
        }
        groq_response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        groq_response.raise_for_status()
        response_data = groq_response.json()
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")

        # Cache the response
        cache[cache_key] = response_text
        logging.info(f"Successfully processed query: {query.query} with model: {query.model}")

        return {"response": response_text}
    except requests.RequestException as e:
        logging.error(f"Groq API error: {str(e)}, URL attempted: {GROQ_API_URL}")
        raise HTTPException(status_code=503, detail=f"Groq API error: {str(e)}")
    except HTTPException as e:
        # Re-raise HTTPException directly to avoid wrapping in 500
        raise e
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
