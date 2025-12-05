# app.py - Flask Chatbot with Stunning UI
from flask import Flask, render_template, request, jsonify, session
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
import hashlib
import time
import secrets
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

class FastScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.timeout = 6
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"@+/#]', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict:
        emails = set()
        phones = set()
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for email in re.findall(email_pattern, text):
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.css', '.js')):
                emails.add(email.lower())
        
        phone_patterns = [
            r'\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}',
            r'\d{4}[\s.-]\d{4}',
            r'\(\d{2,4}\)\s*\d{4}[\s.-]\d{4}',
        ]
        for pattern in phone_patterns:
            for phone in re.findall(pattern, text):
                cleaned = re.sub(r'[^\d+()]', '', phone)
                if 7 <= len(cleaned) <= 20:
                    phones.add(phone.strip())
        
        return {
            "emails": sorted(list(emails))[:5],
            "phones": sorted(list(phones))[:5]
        }
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                tag.decompose()
            
            title = soup.find('title').get_text(strip=True) if soup.find('title') else ""
            
            content = ""
            for selector in ['main', 'article', '[role="main"]', '.main-content', '#main']:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator='\n', strip=True)
                    if len(content) > 200:
                        break
            
            if len(content) < 200:
                texts = []
                for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
                content = '\n'.join(texts)
            
            lines = []
            seen = set()
            for line in content.split('\n'):
                line = self.clean_text(line)
                if len(line) > 25 and line.lower() not in seen:
                    lines.append(line)
                    seen.add(line.lower())
                if len(lines) >= 50:
                    break
            
            content = '\n'.join(lines)
            
            if len(content) < 100:
                return None
            
            return {
                "url": url,
                "title": title[:200],
                "content": content[:4000]
            }
            
        except Exception as e:
            return None
    
    def get_urls_to_scrape(self, base_url: str) -> List[str]:
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
        base_url = base_url.rstrip('/')
        
        paths = ['', '/about', '/about-us', '/services', '/products',
                '/contact', '/contact-us', '/pricing', '/solutions']
        
        urls = [f"{base_url}{path}" for path in paths]
        
        try:
            resp = requests.get(base_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                domain = urlparse(base_url).netloc
                
                for link in soup.find_all('a', href=True)[:60]:
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    if (urlparse(full_url).netloc == domain and 
                        not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png'])):
                        if full_url not in urls:
                            urls.append(full_url)
        except:
            pass
        
        return urls[:50]
    
    def scrape_website(self, base_url: str) -> Tuple[List[Dict], Dict]:
        urls = self.get_urls_to_scrape(base_url)
        pages = []
        all_text = ""
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self.scrape_page, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    if result:
                        pages.append(result)
                        all_text += "\n" + result['content']
                except:
                    pass
        
        contact_info = self.extract_contact_info(all_text)
        
        if len(pages) == 0:
            raise Exception(f"Could not scrape any content from {base_url}")
        
        return pages, contact_info

class SmartAI:
    def __init__(self):
        self.response_cache = {}
        
    def call_llm(self, prompt: str) -> str:
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not set. Please configure OPENROUTER_API_KEY."
        
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        for attempt in range(2):
            try:
                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 400,
                }
                
                resp = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=45)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            self.response_cache[cache_key] = content.strip()
                            return content.strip()
                elif resp.status_code == 429:
                    time.sleep(3)
                    continue
                    
            except:
                if attempt < 1:
                    time.sleep(2)
                    continue
        
        return "I'm having trouble connecting right now. Try asking about contact information!"

class UniversalChatbot:
    def __init__(self, company_name: str, website_url: str):
        self.company_name = company_name
        self.website_url = website_url
        self.pages = []
        self.contact_info = {"emails": [], "phones": []}
        self.ready = False
        self.error = None
        self.ai = SmartAI()
        
    def initialize(self):
        try:
            scraper = FastScraper()
            self.pages, self.contact_info = scraper.scrape_website(self.website_url)
            self.ready = True
            return True
        except Exception as e:
            self.error = str(e)
            return False
    
    def get_context(self, question: str) -> str:
        if not self.pages:
            return ""
        
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        scored_pages = []
        for page in self.pages:
            content_words = set(re.findall(r'\w+', page['content'].lower()))
            score = len(question_words & content_words)
            if score > 0:
                scored_pages.append((score, page))
        
        scored_pages.sort(reverse=True, key=lambda x: x[0])
        
        context_parts = []
        for score, page in scored_pages[:5]:
            context_parts.append(page['content'][:1000])
        
        return "\n\n---\n\n".join(context_parts)
    
    def ask(self, question: str) -> str:
        if not self.ready:
            return "⚠️ Chatbot not initialized yet."
        
        question_lower = question.lower().strip()
        
        greeting_words = ['hi', 'hello', 'hey', 'hai']
        if any(question_lower == g or question_lower.startswith(g + ' ') for g in greeting_words):
            return f"👋 Hi there! I'm the AI assistant for {self.company_name}. How can I help you today?"
        
        contact_keywords = ['email', 'contact', 'phone', 'call', 'reach']
        if any(kw in question_lower for kw in contact_keywords):
            msg = f"📞 Contact Information for {self.company_name}\n\n"
            
            if self.contact_info['emails']:
                msg += "📧 Email:\n" + "\n".join([f"• {e}" for e in self.contact_info['emails']]) + "\n\n"
            
            if self.contact_info['phones']:
                msg += "📱 Phone:\n" + "\n".join([f"• {p}" for p in self.contact_info['phones']]) + "\n\n"
            
            if self.website_url:
                msg += f"🌐 Website: {self.website_url}"
            
            return msg.strip()
        
        context = self.get_context(question)
        
        if not context or len(context) < 50:
            all_content = "\n".join([p['content'][:500] for p in self.pages[:3]])
            if all_content:
                context = all_content
        
        prompt = f"""You are a helpful AI assistant for {self.company_name}.

Based on the following information, answer the user's question clearly.

INFORMATION:
{context[:2500]}

USER QUESTION: {question}

Provide a helpful, conversational answer in 2-4 sentences. Be specific and friendly.

Answer:"""

        return self.ai.call_llm(prompt)

# Flask Routes
@app.route('/')
def index():
    if 'chatbots' not in session:
        session['chatbots'] = {}
    if 'current_company' not in session:
        session['current_company'] = None
    return render_template('index.html')

@app.route('/api/create-chatbot', methods=['POST'])
def create_chatbot():
    data = request.json
    company_name = data.get('company_name')
    website_url = data.get('website_url')
    
    if not company_name or not website_url:
        return jsonify({'success': False, 'error': 'Missing fields'})
    
    slug = re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-')
    
    try:
        chatbot = UniversalChatbot(company_name, website_url)
        success = chatbot.initialize()
        
        if success:
            if 'chatbots' not in session:
                session['chatbots'] = {}
            
            session['chatbots'][slug] = {
                'company_name': chatbot.company_name,
                'website_url': chatbot.website_url,
                'pages_count': len(chatbot.pages),
                'emails': chatbot.contact_info['emails'],
                'phones': chatbot.contact_info['phones']
            }
            session['current_company'] = slug
            session.modified = True
            
            # Store in app context for runtime
            if not hasattr(app, 'chatbots'):
                app.chatbots = {}
            app.chatbots[slug] = chatbot
            
            return jsonify({
                'success': True,
                'slug': slug,
                'data': session['chatbots'][slug]
            })
        else:
            return jsonify({'success': False, 'error': chatbot.error})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chatbots', methods=['GET'])
def get_chatbots():
    if 'chatbots' not in session:
        session['chatbots'] = {}
    return jsonify({
        'chatbots': session['chatbots'],
        'current': session.get('current_company')
    })

@app.route('/api/select-chatbot', methods=['POST'])
def select_chatbot():
    data = request.json
    slug = data.get('slug')
    session['current_company'] = slug
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/delete-chatbot', methods=['POST'])
def delete_chatbot():
    data = request.json
    slug = data.get('slug')
    
    if 'chatbots' in session and slug in session['chatbots']:
        del session['chatbots'][slug]
        
        if hasattr(app, 'chatbots') and slug in app.chatbots:
            del app.chatbots[slug]
        
        if session.get('current_company') == slug:
            session['current_company'] = None
        
        session.modified = True
    
    return jsonify({'success': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    slug = session.get('current_company')
    
    if not slug or not hasattr(app, 'chatbots') or slug not in app.chatbots:
        return jsonify({'success': False, 'error': 'No chatbot selected'})
    
    chatbot = app.chatbots[slug]
    response = chatbot.ask(message)
    
    return jsonify({'success': True, 'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
