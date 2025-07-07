from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import re

# Load environment vars
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Better models for summarization
SUMMARIZATION_MODELS = {
    "primary": "facebook/bart-large-cnn",
    "fallback": "google/pegasus-cnn_dailymail"
}

app = FastAPI()

class ComplaintRequest(BaseModel):
    text: str

def clean_text(text):
    """Remove extra whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?$-]', '', text)
    return text.strip()

def simple_sentence_split(text):
    """Simple sentence splitting without NLTK."""
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_key_info(text):
    """Extract key information like prices, dates, order numbers."""
    info = {}
    
    # Extract prices
    prices = re.findall(r'\$(\d+(?:\.\d{2})?)', text)
    if prices:
        info['prices'] = [float(p) for p in prices]
    
    # Extract order numbers
    order_nums = re.findall(r'#?([A-Z0-9-]{5,})', text)
    if order_nums:
        info['order_numbers'] = order_nums
    
    # Extract time periods
    time_periods = re.findall(r'(\d+)\s*(days?|weeks?|months?|hours?)', text.lower())
    if time_periods:
        info['time_periods'] = time_periods
    
    return info

def detect_sentiment(text):
    """Rule-based sentiment detection."""
    text_lower = text.lower()
    
    # Enhanced rule-based sentiment
    anger_words = ['ridiculous', 'terrible', 'awful', 'angry', 'furious', 'outraged', 'unacceptable', 'disgusting']
    frustration_words = ['frustrated', 'disappointed', 'annoyed', 'upset', 'bothered', 'poor', 'slow', 'late', 'broken', 'damaged']
    polite_words = ['please', 'thank you', 'appreciate', 'understand', 'hope', 'wondering', 'grateful']
    
    if any(word in text_lower for word in anger_words):
        return "angry"
    elif any(word in text_lower for word in frustration_words):
        return "frustrated"
    elif any(word in text_lower for word in polite_words):
        return "polite"
    else:
        return "neutral"

def categorize_complaint(text):
    """Categorize the complaint type."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['shipping', 'delivery', 'arrived', 'package', 'delayed']):
        return "shipping"
    elif any(word in text_lower for word in ['charge', 'billing', 'payment', 'refund', 'money']):
        return "billing"
    elif any(word in text_lower for word in ['broken', 'defective', 'quality', 'damaged', 'stopped working']):
        return "product_defect"
    elif any(word in text_lower for word in ['wrong', 'incorrect', 'mistake', 'different']):
        return "wrong_item"
    elif any(word in text_lower for word in ['customer service', 'support', 'representative', 'staff']):
        return "service"
    elif any(word in text_lower for word in ['website', 'app', 'technical', 'crash']):
        return "technical"
    elif any(word in text_lower for word in ['price', 'cost', 'pricing']):
        return "pricing"
    else:
        return "general"

def create_custom_summary(text, key_info, category):
    """Create a custom summary based on complaint type and extracted info."""
    
    templates = {
        "shipping": "Customer experienced shipping issues",
        "billing": "Customer reports billing problem",
        "product_defect": "Customer's product is defective/damaged",
        "wrong_item": "Customer received incorrect item",
        "service": "Customer had poor service experience",
        "technical": "Customer experiencing technical issues",
        "pricing": "Customer concerned about pricing",
        "general": "Customer complaint"
    }
    
    base_summary = templates.get(category, "Customer complaint")
    details = []
    
    if 'prices' in key_info:
        if len(key_info['prices']) > 1:
            details.append(f"involving amounts from ${min(key_info['prices'])} to ${max(key_info['prices'])}")
        else:
            details.append(f"involving ${key_info['prices'][0]}")
    
    if 'time_periods' in key_info:
        time_detail = key_info['time_periods'][0]
        details.append(f"over {time_detail[0]} {time_detail[1]}")
    
    if 'order_numbers' in key_info:
        details.append(f"for order {key_info['order_numbers'][0]}")
    
    text_lower = text.lower()
    if 'want refund' in text_lower or 'full refund' in text_lower:
        details.append("Customer requests refund")
    elif 'replacement' in text_lower:
        details.append("Customer seeks replacement")
    elif 'cancel' in text_lower:
        details.append("Customer wants to cancel")
    
    if details:
        summary = f"{base_summary} {', '.join(details[:2])}"
    else:
        summary = base_summary
    
    return summary

async def get_ai_summary(text):
    """Get AI-generated summary."""
    
    for model_name, model_id in SUMMARIZATION_MODELS.items():
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            payload = {
                "inputs": text,
                "parameters": {
                    "min_length": 15,
                    "max_length": 50,
                    "do_sample": False,
                    "length_penalty": 2.0,
                    "num_beams": 4
                }
            }
            
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                output = response.json()
                
                if isinstance(output, list) and len(output) > 0:
                    if 'summary_text' in output[0]:
                        return output[0]['summary_text']
                elif isinstance(output, dict):
                    if 'summary_text' in output:
                        return output['summary_text']
                
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    return None

@app.post("/summarize")
async def summarize(complaint: ComplaintRequest):
    original_text = clean_text(complaint.text)
    original_word_count = len(original_text.split())

    if original_word_count < 3:
        return {
            "success": False,
            "error": "Text too short to summarize"
        }
    
    key_info = extract_key_info(original_text)
    category = categorize_complaint(original_text)
    sentiment = detect_sentiment(original_text)
    
    summary = None
    
    if original_word_count < 20:
        summary = create_custom_summary(original_text, key_info, category)
    else:
        summary = await get_ai_summary(original_text)
        
        if not summary:
            summary = create_custom_summary(original_text, key_info, category)
    
    summary = clean_text(summary)
    
    if summary == original_text[:len(summary)]:
        summary = create_custom_summary(original_text, key_info, category)
    
    if len(summary.split()) >= original_word_count:
        sentences = simple_sentence_split(summary)
        summary = sentences[0] if sentences else summary[:100] + "..."
    
    summary_word_count = len(summary.split())

    return {
        "success": True,
        "original_text": original_text,
        "original_word_count": original_word_count,
        "summary": summary,
        "summary_word_count": summary_word_count,
        "sentiment": sentiment,
        "category": category,
        "key_info": key_info
    }

@app.post("/respond")
async def respond(complaint: ComplaintRequest):
    cleaned = clean_text(complaint.text)
    
    key_info = extract_key_info(cleaned)
    category = categorize_complaint(cleaned)
    sentiment = detect_sentiment(cleaned)
    
    if sentiment == "angry":
        response_start = "We sincerely apologize for your frustrating experience and understand your anger."
    elif sentiment == "frustrated":
        response_start = "We're sorry to hear about your disappointing experience."
    elif sentiment == "polite":
        response_start = "Thank you for bringing this to our attention in such a courteous manner."
    else:
        response_start = "Thank you for your feedback."
    
    category_responses = {
        "shipping": "We're investigating the shipping delay and will provide you with an update within 24 hours.",
        "billing": "We're reviewing your billing concern and will correct any errors immediately.",
        "product_defect": "We'll arrange for a replacement or refund for your defective product.",
        "wrong_item": "We'll send you the correct item and arrange pickup of the incorrect one.",
        "service": "We're addressing this service issue with our team and will follow up with you.",
        "technical": "Our technical team is working to resolve this issue as quickly as possible.",
        "pricing": "We're reviewing your pricing concern and will provide clarification.",
        "general": "We're looking into your concern and will respond with a resolution soon."
    }
    
    category_response = category_responses.get(category, "We're looking into your concern.")
    
    reference = ""
    if 'order_numbers' in key_info:
        reference = f" Reference: {key_info['order_numbers'][0]}"
    
    full_response = f"{response_start} {category_response}{reference}"
    
    return {
        "success": True,
        "response": full_response,
        "sentiment": sentiment,
        "category": category
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "complaint-summarizer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)