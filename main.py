# main.py - Part 1: Backend with Web Scraping

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import re

load_dotenv()

app = FastAPI(title="IT Company Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL_NAME = os.getenv("MODEL_NAME", "kwaipilot/kat-coder-pro:free")
COMPANY_WEBSITE = os.getenv("COMPANY_WEBSITE", "https://yourcompany.com")

# Priority pages to scrape
PRIORITY_PAGES = [
    "", "about", "services", "solutions", "products", "contact", "team",
    "careers", "blog", "case-studies", "portfolio", "industries",
    "technology", "expertise", "what-we-do", "who-we-are"
]

# Global variables
scraped_data = {"content": "", "company_info": {}}
chroma_client = None
collection = None
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Web Scraper Class
class WebScraper:
    def __init__(self):
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'addresses': []
        }

    def clean_text(self, text):
        text = ' '.join(text.split())
        return re.sub(r'\s+', ' ', text).strip()

    def extract_contact_info(self, text):
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.svg')):
                self.company_info['emails'].add(email.lower())

        # Extract phone numbers
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if 10 <= len(cleaned) <= 15:
                    self.company_info['phones'].add(phone.strip())

        # Extract addresses (look for common patterns)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Look for lines with postal codes or common address keywords
            if re.search(r'\d{5,6}', line) or any(word in line.lower() for word in ['street', 'road', 'avenue', 'city', 'state']):
                if 20 < len(line) < 200:
                    self.company_info['addresses'].append(self.clean_text(line))

    def is_valid_url(self, url, base_domain):
        try:
            parsed = urlparse(url)
            if parsed.netloc != base_domain:
                return False
            skip_patterns = [
                r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$',
                r'/wp-admin/', r'/wp-includes/', r'/login', r'/register',
                r'/cart/', r'/checkout/', r'/feed/', r'/rss/'
            ]
            return not any(re.search(pattern, url.lower()) for pattern in skip_patterns)
        except:
            return False

    def extract_content(self, soup, url):
        content_dict = {'url': url, 'title': '', 'content': ''}

        # Get title
        title = soup.find('title')
        if title:
            content_dict['title'] = title.get_text(strip=True)

        # Get meta description
        meta = soup.find('meta', attrs={"name": "description"})
        if meta and meta.get("content"):
            content_dict['meta_description'] = meta["content"]

        # Extract contact info from full page text
        full_text = soup.get_text(separator="\n", strip=True)
        self.extract_contact_info(full_text)

        # Remove unwanted tags
        for tag in soup(['script', 'style', 'nav', 'aside', 'iframe', 'noscript', 'form', 'header', 'footer']):
            tag.decompose()

        # Try to find main content
        main_content = None
        content_selectors = ["main", "article", "[role='main']", ".content", ".main-content", "#content"]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find("body")

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]
            content_dict['content'] = "\n".join(lines)

        return content_dict

    def scrape_website(self, base_url, max_pages=40):
        visited = set()
        all_content = []
        queue = deque()
        base_domain = urlparse(base_url).netloc

        # Add priority pages first
        for page in PRIORITY_PAGES:
            queue.append(urljoin(base_url, page))

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        print(f"Starting to scrape {base_url}...")

        while queue and len(visited) < max_pages:
            url = queue.popleft().split("#")[0].split("?")[0]

            if url in visited or not self.is_valid_url(url, base_domain):
                continue

            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    continue

                visited.add(url)
                print(f"Scraped {len(visited)}/{max_pages}: {url}")

                soup = BeautifulSoup(response.text, "html.parser")
                data = self.extract_content(soup, url)

                if len(data['content']) > 100:
                    formatted = f"URL: {data['url']}\nTITLE: {data['title']}\n"
                    if 'meta_description' in data:
                        formatted += f"DESCRIPTION: {data['meta_description']}\n"
                    formatted += f"\nCONTENT:\n{data['content']}"
                    all_content.append(formatted)

                # Find more links
                for link in soup.find_all("a", href=True):
                    next_url = urljoin(url, link['href']).split("#")[0].split("?")[0]
                    if next_url not in visited and self.is_valid_url(next_url, base_domain):
                        queue.append(next_url)

            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                continue

        print(f"Scraping complete! Visited {len(visited)} pages.")

        # Add contact information at the beginning
        if self.company_info['emails'] or self.company_info['phones']:
            header = "COMPANY CONTACT INFORMATION\n" + "="*50 + "\n\n"
            
            if self.company_info['emails']:
                header += "Emails:\n" + "\n".join([f"• {e}" for e in sorted(self.company_info['emails'])]) + "\n\n"
            
            if self.company_info['phones']:
                header += "Phones:\n" + "\n".join([f"• {p}" for p in sorted(self.company_info['phones'])]) + "\n\n"
            
            if self.company_info['addresses']:
                header += "Addresses:\n" + "\n".join([f"• {a}" for a in self.company_info['addresses'][:3]]) + "\n"
            
            all_content.insert(0, header)

        return "\n\n" + "="*80 + "\n\n".join(all_content), self.company_info

# MySQL Connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "chatbot_db")
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Initialize database
def init_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_contacts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                phone VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session (session_id),
                INDEX idx_email (email)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session (session_id)
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database initialized successfully")

init_database()

# Initialize ChromaDB with scraped content
def initialize_knowledge_base():
    global chroma_client, collection, scraped_data
    
    print("🌐 Starting web scraping...")
    scraper = WebScraper()
    content, company_info = scraper.scrape_website(COMPANY_WEBSITE, max_pages=40)
    
    if len(content) < 1000:
        print("❌ Insufficient content scraped")
        return False
    
    scraped_data['content'] = content
    scraped_data['company_info'] = company_info
    
    print("📦 Initializing ChromaDB...")
    chroma_client = chromadb.Client()
    
    try:
        collection = chroma_client.get_collection(name="company_knowledge")
        print("✅ Using existing collection")
    except:
        # Split content into chunks
        chunks = []
        paragraphs = content.split("\n\n")
        
        for para in paragraphs:
            if len(para) > 100:  # Only include substantial paragraphs
                # Split large paragraphs
                if len(para) > 1000:
                    words = para.split()
                    for i in range(0, len(words), 200):
                        chunk = " ".join(words[i:i+200])
                        if len(chunk) > 100:
                            chunks.append(chunk)
                else:
                    chunks.append(para)
        
        print(f"📝 Created {len(chunks)} text chunks")
        
        collection = chroma_client.create_collection(
            name="company_knowledge",
            embedding_function=embedding_function
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            collection.add(
                documents=batch,
                ids=[f"doc_{j}" for j in range(i, i+len(batch))]
            )
        
        print("✅ Knowledge base initialized!")
    
    return True

# Initialize knowledge base on startup
@app.on_event("startup")
async def startup_event():
    initialize_knowledge_base()

# Session store
sessions = {}

# Pydantic models
class ChatMessage(BaseModel):
    session_id: str
    message: str

class UserInfo(BaseModel):
    session_id: str
    name: str
    email: str
    phone: str

class ChatResponse(BaseModel):
    response: str
    question_count: int
    needs_user_info: bool

# Query knowledge base
def query_knowledge_base(query: str, n_results: int = 3):
    if not collection:
        return ""
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results['documents']:
        return "\n\n".join(results['documents'][0])
    return ""

# Generate LLM response
def generate_response(user_message: str, context: str):
    system_prompt = f"""You are a helpful AI assistant for this company's website.

Use the following information to answer questions accurately:
{context}

Instructions:
- Be friendly, professional, and concise
- Answer ONLY using the provided context
- If information is not in the context, say you'll connect them with the team
- Keep responses under 100 words
- Do not make up information"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return "I'm having trouble responding right now. Please try again."

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    session_id = message.session_id
    user_message = message.message
    
    if session_id not in sessions:
        sessions[session_id] = {
            "question_count": 0,
            "user_info_collected": False
        }
    
    session = sessions[session_id]
    
    # Check for contact info request
    contact_keywords = ["email", "contact", "phone", "address", "office", "location", "reach"]
    if any(keyword in user_message.lower() for keyword in contact_keywords):
        info = scraped_data.get('company_info', {})
        response_text = "📞 **Contact Information:**\n\n"
        
        if info.get('emails'):
            response_text += "📧 **Emails:**\n" + "\n".join([f"• {e}" for e in list(info['emails'])[:3]]) + "\n\n"
        if info.get('phones'):
            response_text += "☎️ **Phones:**\n" + "\n".join([f"• {p}" for p in list(info['phones'])[:3]]) + "\n\n"
        if info.get('addresses'):
            response_text += "📍 **Address:**\n" + "\n".join([f"• {a}" for a in info['addresses'][:2]])
        
        session["question_count"] += 1
        
        return ChatResponse(
            response=response_text if response_text.strip() != "📞 **Contact Information:**" else "Contact information will be provided by our team.",
            question_count=session["question_count"],
            needs_user_info=False
        )
    
    # Check if we need user info
    if session["question_count"] >= 3 and not session["user_info_collected"]:
        return ChatResponse(
            response="I'd love to help you further! Before we continue, could you please share your name, email, and phone number? This helps us provide you with better personalized assistance.",
            question_count=session["question_count"],
            needs_user_info=True
        )
    
    # Query knowledge base
    context = query_knowledge_base(user_message, n_results=3)
    
    # Generate response
    bot_response = generate_response(user_message, context)
    
    # Save to database
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (session_id, question, answer) VALUES (%s, %s, %s)",
                (session_id, user_message, bot_response)
            )
            conn.commit()
            cursor.close()
        except:
            pass
        finally:
            conn.close()
    
    session["question_count"] += 1
    
    return ChatResponse(
        response=bot_response,
        question_count=session["question_count"],
        needs_user_info=False
    )

@app.post("/submit-info")
async def submit_user_info(user_info: UserInfo):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO user_contacts (session_id, name, email, phone)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (
            user_info.session_id,
            user_info.name,
            user_info.email,
            user_info.phone
        ))
        conn.commit()
        cursor.close()
        conn.close()
        
        if user_info.session_id in sessions:
            sessions[user_info.session_id]["user_info_collected"] = True
        
        return {
            "message": f"Thank you {user_info.name}! Your information has been saved. Our team will contact you at {user_info.email}. How else can I help you?",
            "success": True
        }
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Chatbot API is running!", "status": "ready"}

# CONTINUE TO PART 2 FOR CHAT WIDGET HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
