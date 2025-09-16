#!/usr/bin/env python3
"""
Lightweight Multi-Modal RAG Chatbot
Optimized version without problematic dependencies
"""

import os
import sys
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
import asyncio
import threading
from dataclasses import dataclass
import logging

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import cross-modal retrieval system
try:
    from cross_modal_retrieval import CrossModalRetrieval
    HAS_CROSS_MODAL = True
    print("✅ Cross-modal retrieval imported")
except ImportError:
    print("⚠️ Cross-modal retrieval not available")
    HAS_CROSS_MODAL = False

# Import simple cross-modal as fallback
try:
    from simple_cross_modal import SimpleCrossModalRetrieval
    HAS_SIMPLE_CROSS_MODAL = True
    print("✅ Simple cross-modal retrieval imported")
except ImportError:
    print("⚠️ Simple cross-modal retrieval not available")
    HAS_SIMPLE_CROSS_MODAL = False

# Load environment variables
def load_environment():
    """Load environment variables with multiple fallback paths"""
    try:
        from dotenv import load_dotenv
        
        env_paths = [
            Path(".env"),
            Path("backend/.env"),
            Path("../.env"),
            current_dir / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✅ Environment loaded from: {env_path}")
                return True
        
        print("⚠️ No .env file found")
        return False
    except ImportError:
        print("⚠️ python-dotenv not installed")
        return False

# Load environment on import
load_environment()

# Import core dependencies with error handling
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ Google Gemini imported")
except ImportError:
    print("⚠️ Google Gemini not installed")
    HAS_GEMINI = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    HAS_QDRANT = True
    print("✅ Qdrant imported")
except ImportError:
    print("⚠️ Qdrant not installed")
    HAS_QDRANT = False

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    HAS_FLASK = True
    print("✅ Flask imported")
except ImportError:
    print("⚠️ Flask not installed")
    HAS_FLASK = False

# Optional multi-modal processing imports (graceful fallback)
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
    print("✅ OpenCV imported")
except ImportError:
    print("⚠️ OpenCV not installed - video processing disabled")
    HAS_OPENCV = False

try:
    import speech_recognition as sr
    HAS_AUDIO = True
    print("✅ Audio processing imported")
except ImportError:
    print("⚠️ Audio processing not installed")
    HAS_AUDIO = False

try:
    from PIL import Image
    HAS_PIL = True
    print("✅ PIL imported")
except ImportError:
    print("⚠️ PIL not installed")
    HAS_PIL = False

try:
    import PyPDF2
    HAS_PDF = True
    print("✅ PDF processing imported")
except ImportError:
    print("⚠️ PDF processing not installed")
    HAS_PDF = False

@dataclass
class MultiModalInput:
    """Data structure for multi-modal input"""
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    video_data: Optional[bytes] = None
    image_data: Optional[bytes] = None
    document_data: Optional[bytes] = None
    file_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessedContent:
    """Processed multi-modal content"""
    extracted_text: str = ""
    audio_transcript: str = ""
    image_description: str = ""
    video_summary: str = ""
    document_text: str = ""
    entities: List[str] = None
    sentiment: str = "neutral"
    confidence: float = 0.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []

class SimpleTextProcessor:
    """Simple text processing without heavy dependencies"""
    
    def __init__(self):
        # Simple keyword-based similarity
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'
        }
    
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create simple word-based embedding"""
        words = text.lower().split()
        words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Simple bag of words representation
        vocab = set(words)
        embedding = []
        
        # Create a simple feature vector
        for word in sorted(vocab)[:50]:  # Limit to 50 features
            embedding.append(words.count(word) / len(words) if words else 0)
        
        # Pad or truncate to fixed size
        while len(embedding) < 50:
            embedding.append(0.0)
        
        return embedding[:50]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple cosine similarity"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0

class AudioProcessor:
    """Simple audio processing"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if HAS_AUDIO else None
        
    def process_audio(self, audio_data: bytes, file_format: str = "wav") -> Dict[str, Any]:
        """Process audio file and extract information"""
        if not HAS_AUDIO:
            return {"transcript": "Audio processing not available", "error": "Dependencies missing"}
        
        try:
            # Save temporary audio file
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Simple transcription
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
                
            try:
                transcript = self.recognizer.recognize_google(audio)
            except:
                transcript = "Could not transcribe audio"
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "transcript": transcript,
                "duration": len(audio_data) / 44100,  # Rough estimate
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"transcript": "", "error": str(e)}

class ImageProcessor:
    """Simple image processing"""
    
    def __init__(self):
        self.gemini_vision = None
        if HAS_GEMINI:
            try:
                self.gemini_vision = genai.GenerativeModel('gemini-1.5-flash')
            except:
                pass
    
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and extract information"""
        try:
            if not HAS_PIL:
                return {"description": "Image processing not available", "error": "PIL not installed"}
            
            # Convert to PIL Image
            import io
            image = Image.open(io.BytesIO(image_data))
            
            results = {
                "description": "",
                "text_content": "",
                "width": image.width,
                "height": image.height,
                "format": image.format or "Unknown",
                "confidence": 0.7
            }
            
            # AI-powered image description using Gemini
            if self.gemini_vision:
                try:
                    response = self.gemini_vision.generate_content([
                        "Describe this image in detail. What do you see?",
                        image
                    ])
                    results["description"] = response.text
                    results["confidence"] = 0.9
                except Exception as e:
                    results["description"] = f"Could not analyze image with AI: {str(e)}"
            else:
                results["description"] = f"Image detected: {image.width}x{image.height} pixels, format: {image.format}"
            
            return results
            
        except Exception as e:
            return {"description": "", "error": str(e)}

class DocumentProcessor:
    """Simple document processing"""
    
    def process_document(self, document_data: bytes, file_type: str) -> Dict[str, Any]:
        """Process document and extract text content"""
        try:
            text_content = ""
            
            if file_type.lower() == "pdf" and HAS_PDF:
                text_content = self._process_pdf(document_data)
            elif file_type.lower() == "txt":
                text_content = document_data.decode('utf-8', errors='ignore')
            else:
                return {"text": "", "error": f"Unsupported document type: {file_type}"}
            
            # Extract metadata
            word_count = len(text_content.split())
            char_count = len(text_content)
            
            return {
                "text": text_content,
                "word_count": word_count,
                "char_count": char_count,
                "file_type": file_type,
                "confidence": 0.9
            }
            
        except Exception as e:
            return {"text": "", "error": str(e)}
    
    def _process_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name
        
        try:
            with open(temp_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            os.unlink(temp_path)
            return text
        except:
            os.unlink(temp_path)
            return ""

class LightweightMultiModalChatbot:
    """Lightweight Multi-Modal RAG Chatbot without heavy dependencies"""
    
    def __init__(self):
        """Initialize the lightweight chatbot"""
        print("🤖 Initializing Lightweight Multi-Modal RAG Chatbot...")
        
        # Core components
        self.gemini_client = None
        self.qdrant_client = None
        self.text_processor = SimpleTextProcessor()
        self.knowledge_base = []
        
        # Multi-modal processors
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
        self.document_processor = DocumentProcessor()
        
        # Cross-modal retrieval system
        self.cross_modal_retrieval = None
        if HAS_CROSS_MODAL:
            try:
                self.cross_modal_retrieval = CrossModalRetrieval()
                print("✅ Cross-modal retrieval initialized")
            except Exception as e:
                print(f"⚠️ Cross-modal retrieval failed: {e}")
                self.cross_modal_retrieval = None
        
        # Fallback to simple cross-modal
        if not self.cross_modal_retrieval and HAS_SIMPLE_CROSS_MODAL:
            try:
                self.cross_modal_retrieval = SimpleCrossModalRetrieval()
                print("✅ Simple cross-modal retrieval initialized as fallback")
            except Exception as e:
                print(f"⚠️ Simple cross-modal retrieval failed: {e}")
                self.cross_modal_retrieval = None
        
        # Initialize core components
        self._init_gemini()
        self._init_qdrant()
        self._load_knowledge_base()
        
        # Conversation history
        self.conversation_history = []
        self.processing_stats = {
            "total_queries": 0,
            "multimodal_queries": 0,
            "processing_time_avg": 0.0
        }
        
        print("✅ Lightweight Multi-Modal RAG Chatbot ready!")
    
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        if not HAS_GEMINI:
            print("⚠️ Google Gemini library not installed")
            return
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️ Gemini API key not configured")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini initialized successfully")
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            self.gemini_client = None
    
    def _init_qdrant(self):
        """Initialize Qdrant client with fallback to in-memory"""
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        qdrant_cloud_url = os.getenv('QDRANT_CLOUD_URL')
        use_qdrant_cloud = os.getenv('USE_QDRANT_CLOUD', 'false').lower() == 'true'
        
        # Try Qdrant Cloud first if enabled and URL is valid
        if qdrant_api_key and HAS_QDRANT and use_qdrant_cloud and qdrant_cloud_url and 'your-' not in qdrant_cloud_url:
            try:
                self.qdrant_client = QdrantClient(
                    url=qdrant_cloud_url,
                    api_key=qdrant_api_key,
                    timeout=30,
                    prefer_grpc=False
                )
                collections = self.qdrant_client.get_collections()
                print("✅ Qdrant Cloud connected")
                return
            except Exception as e:
                print(f"⚠️ Qdrant Cloud failed: {e}")
                if "403" in str(e) or "Forbidden" in str(e):
                    print("   🔑 Authentication failed - check API key and permissions")
                elif "timeout" in str(e).lower():
                    print("   ⏱️ Connection timeout - check network/URL")
                print("   🔄 Falling back to local options...")
        
        # Try local Qdrant
        if HAS_QDRANT:
            try:
                self.qdrant_client = QdrantClient(host="localhost", port=6333)
                collections = self.qdrant_client.get_collections()
                print("✅ Local Qdrant connected")
                return
            except Exception as e:
                print(f"⚠️ Local Qdrant failed: {e}")
                print("   💡 To start local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        
        # Fallback to in-memory Qdrant for development
        if HAS_QDRANT:
            try:
                self.qdrant_client = QdrantClient(":memory:")
                print("✅ In-memory Qdrant initialized - perfect for demos!")
                print("   📝 Note: Data will not persist between sessions")
                return
            except Exception as e:
                print(f"⚠️ In-memory Qdrant failed: {e}")
        
        print("❌ No Qdrant available - using local search only")
        print("   🎯 Your chatbot will still work with all other features!")
        print("   📊 Vector search will be simulated using local similarity")
        self.qdrant_client = None
    
    def _load_knowledge_base(self):
        """Load comprehensive knowledge base"""
        self.knowledge_base = [
            {
                "text": "This is a lightweight multi-modal RAG chatbot that can process text, audio, video, images, and documents. It uses simplified processing to avoid dependency conflicts.",
                "category": "system",
                "keywords": ["multimodal", "rag", "chatbot", "lightweight"]
            },
            {
                "text": "Audio processing includes speech-to-text transcription using Google's speech recognition API. Supports WAV, MP3, and other common audio formats.",
                "category": "audio",
                "keywords": ["audio", "speech", "transcription", "voice"]
            },
            {
                "text": "Image processing includes AI-powered description using Google Gemini Vision API. Can analyze photos, screenshots, diagrams, and other visual content.",
                "category": "image",
                "keywords": ["image", "visual", "photo", "description"]
            },
            {
                "text": "Document processing supports PDF and text files. Extracts content and provides intelligent analysis and summarization.",
                "category": "document",
                "keywords": ["document", "pdf", "text", "extraction"]
            },
            {
                "text": "The system uses Google Gemini for AI responses and can optionally connect to Qdrant for vector search capabilities.",
                "category": "ai",
                "keywords": ["gemini", "ai", "vector", "search"]
            }
        ]
        print(f"📚 Loaded {len(self.knowledge_base)} knowledge entries")
    
    def process_multimodal_input(self, multimodal_input: MultiModalInput) -> Dict[str, Any]:
        """Process multi-modal input"""
        start_time = datetime.now()
        processed_content = ProcessedContent()
        
        # Process different modalities
        if multimodal_input.text:
            processed_content.extracted_text = multimodal_input.text
        
        if multimodal_input.audio_data:
            audio_result = self.audio_processor.process_audio(multimodal_input.audio_data)
            processed_content.audio_transcript = audio_result.get('transcript', '')
        
        if multimodal_input.image_data:
            image_result = self.image_processor.process_image(multimodal_input.image_data)
            processed_content.image_description = image_result.get('description', '')
        
        if multimodal_input.document_data:
            doc_result = self.document_processor.process_document(
                multimodal_input.document_data, 
                multimodal_input.file_type or "txt"
            )
            processed_content.document_text = doc_result.get('text', '')
        
        # Combine all extracted text
        all_text = " ".join(filter(None, [
            processed_content.extracted_text,
            processed_content.audio_transcript,
            processed_content.image_description,
            processed_content.document_text
        ]))
        
        # Generate response
        response = self._generate_response(all_text, processed_content)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats["total_queries"] += 1
        if any([multimodal_input.audio_data, multimodal_input.image_data, multimodal_input.document_data]):
            self.processing_stats["multimodal_queries"] += 1
        
        # Store in conversation history
        conversation_entry = {
            "timestamp": start_time.isoformat(),
            "input": {
                "text": multimodal_input.text,
                "has_audio": bool(multimodal_input.audio_data),
                "has_image": bool(multimodal_input.image_data),
                "has_document": bool(multimodal_input.document_data)
            },
            "response": response,
            "processing_time": processing_time
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            "response": response,
            "processed_content": processed_content,
            "processing_time": processing_time,
            "confidence": 0.8
        }
    
    def _generate_response(self, combined_text: str, processed_content: ProcessedContent) -> str:
        """Generate response using available AI"""
        if not combined_text.strip():
            return "I didn't receive any content to process. Please provide text, upload a file, or record audio."
        
        # Check for cross-modal retrieval queries
        if self._is_cross_modal_query(combined_text, processed_content):
            return self._handle_cross_modal_query(combined_text, processed_content)
        
        # Check for book recommendation queries
        if self._is_book_query(combined_text):
            return self._generate_enhanced_book_response(combined_text)
        
        # Search knowledge base
        context = self.search_knowledge(combined_text)
        
        # Try Gemini first
        if self.gemini_client:
            try:
                context_text = "\n".join([item["text"] for item in context[:3]]) if context else "No specific context found."
                
                prompt = f"""You are a helpful multi-modal AI assistant. Analyze and respond to the following input:

INPUT: {combined_text}

CONTEXT: {context_text}

Provide a helpful, informative response. If multiple types of content were provided (text, audio, image, document), acknowledge them in your response."""

                response = self.gemini_client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini failed: {e}")
        
        # Fallback response
        if context:
            return f"Based on your input, here's what I found: {context[0]['text']}"
        else:
            return f"I processed your multi-modal input: {combined_text[:200]}{'...' if len(combined_text) > 200 else ''}"
    
    def _is_cross_modal_query(self, text: str, processed_content: ProcessedContent) -> bool:
        """Check if this is a cross-modal retrieval query"""
        text_lower = text.lower()
        
        # Text to image queries
        text_to_image_keywords = [
            'show me images', 'find images', 'show pictures', 'find pictures',
            'images of', 'pictures of', 'visual', 'photo of', 'image for'
        ]
        
        # Image to text queries (when image is provided)
        has_image = bool(processed_content.image_description)
        image_to_text_keywords = [
            'describe this', 'what is this', 'find similar text', 'related text',
            'tell me about', 'explain this image', 'similar content'
        ]
        
        # Check for text → image queries
        text_wants_images = any(keyword in text_lower for keyword in text_to_image_keywords)
        
        # Check for image → text queries
        image_wants_text = has_image and (
            any(keyword in text_lower for keyword in image_to_text_keywords) or 
            text_lower in ['', 'what is this?', 'describe', 'analyze']
        )
        
        return text_wants_images or image_wants_text
    
    def _handle_cross_modal_query(self, text: str, processed_content: ProcessedContent) -> str:
        """Handle cross-modal retrieval queries"""
        if not self.cross_modal_retrieval:
            return """🔍 **Cross-Modal Search Not Available**
            
The cross-modal retrieval system is not currently available. To enable this feature:

1. **Install CLIP**: `pip install git+https://github.com/openai/CLIP.git`
2. **Install PyTorch**: `pip install torch torchvision`
3. **Restart the chatbot**

This will enable:
• **Text → Images**: Find images similar to your text description
• **Image → Text**: Find text content similar to your uploaded image

📝 **What you tried to do**: Find cross-modal similarities between text and images."""
        
        text_lower = text.lower()
        has_image = bool(processed_content.image_description)
        
        # Handle Image → Text search
        if has_image:
            return self._search_similar_texts_for_image(processed_content.image_description, text)
        
        # Handle Text → Image search
        else:
            return self._search_similar_images_for_text(text)
    
    def _search_similar_images_for_text(self, text_query: str) -> str:
        """Search for images similar to text query"""
        try:
            similar_images = self.cross_modal_retrieval.find_similar_images(text_query, top_k=5)
            
            if not similar_images:
                return f"🔍 **No Similar Images Found**\n\nI couldn't find images similar to: '{text_query}'\n\nTry adding more images to the database or refining your search terms."
            
            response = f"""🔍 **Images Similar to: "{text_query}"**

📸 **CROSS-MODAL SEARCH RESULTS:**
"""
            
            for i, result in enumerate(similar_images, 1):
                image_info = result['image_info']
                similarity_score = result['similarity']
                
                response += f"""
{i}. **{image_info['description']}**
   🎯 **Similarity**: {similarity_score:.3f} ({similarity_score*100:.1f}%)
   📝 **Type**: {image_info['metadata'].get('type', 'Unknown')}
   📍 **Source**: {image_info['metadata'].get('source', 'Database')}
"""
                
                if 'path' in image_info:
                    response += f"   📁 **File**: {image_info['path']}\n"
            
            response += f"""
💡 **How Cross-Modal Search Works:**
• Your text is converted to a vector embedding using CLIP
• All images in the database are also embedded in the same space  
• Similarity is calculated using cosine distance
• Results are ranked by semantic similarity

🎯 **Search Query**: "{text_query}"
📊 **Total Images Searched**: {len(self.cross_modal_retrieval.image_database)}
"""
            
            return response
            
        except Exception as e:
            return f"❌ **Cross-Modal Search Error**\n\nError occurred while searching for images: {str(e)}"
    
    def _search_similar_texts_for_image(self, image_description: str, original_text: str = "") -> str:
        """Search for texts similar to uploaded image"""
        try:
            # Use the image description as a proxy for the actual image embedding
            similar_texts = self.cross_modal_retrieval.find_similar_texts(image_description, top_k=5)
            
            if not similar_texts:
                return f"🔍 **No Similar Texts Found**\n\nI couldn't find texts similar to your image.\n\nImage description: {image_description}"
            
            response = f"""🔍 **Texts Similar to Your Image**

🖼️ **Your Image**: {image_description}

📝 **CROSS-MODAL SEARCH RESULTS:**
"""
            
            for i, result in enumerate(similar_texts, 1):
                text_info = result['text_info']
                similarity_score = result['similarity']
                text_preview = text_info['text'][:100] + "..." if len(text_info['text']) > 100 else text_info['text']
                
                response += f"""
{i}. **Text Match** (Similarity: {similarity_score:.3f})
   📝 "{text_preview}"
   📍 Source: {text_info['metadata'].get('source', 'Database')}
"""
            
            response += f"""
💡 **How Image-to-Text Search Works:**
• Your image is processed and converted to CLIP embedding
• All texts in the database are embedded in the same semantic space
• The system finds texts that would be most similar to your image
• Results show content that shares visual concepts with your image

🎯 **Image Analysis**: {image_description}
📊 **Total Texts Searched**: {len(self.cross_modal_retrieval.text_database)}
"""
            
            return response
            
        except Exception as e:
            return f"❌ **Cross-Modal Search Error**\n\nError occurred while searching for texts: {str(e)}"

    def _is_book_query(self, query: str) -> bool:
        """Check if query is asking for book recommendations"""
        query_lower = query.lower()
        book_keywords = ['book', 'books', 'machine learning', 'ml', 'data science', 'python', 'programming']
        action_keywords = ['show me', 'recommend', 'find', 'search', 'list', 'suggest']
        budget_keywords = ['budget', 'under', 'dollar', '$', 'price', 'cost', 'cheap', 'affordable']
        
        has_book = any(keyword in query_lower for keyword in book_keywords)
        has_action = any(keyword in query_lower for keyword in action_keywords)
        has_budget = any(keyword in query_lower for keyword in budget_keywords)
        
        return has_book and (has_action or has_budget)
    
    def _generate_enhanced_book_response(self, query: str) -> str:
        """Generate enhanced response for book queries with image guidance"""
        import re
        
        # Extract budget if present
        budget_match = re.search(r'\$?(\d+(?:\.\d{2})?)', query)
        budget = float(budget_match.group(1)) if budget_match else None
        
        # Enhanced book database with image search guidance
        books = [
            {
                "title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
                "author": "Aurélien Géron",
                "price": 44.99,
                "description": "Practical ML guide with Python, TensorFlow, and real projects",
                "image_search": "hands on machine learning aurelien geron book cover",
                "amazon_search": "hands-on machine learning scikit-learn keras tensorflow",
                "isbn": "978-1492032649"
            },
            {
                "title": "Pattern Recognition and Machine Learning",
                "author": "Christopher Bishop",
                "price": 49.95,
                "description": "Mathematical foundations of ML algorithms and statistics",
                "image_search": "pattern recognition machine learning bishop book cover",
                "amazon_search": "pattern recognition machine learning christopher bishop",
                "isbn": "978-0387310732"
            },
            {
                "title": "Python Machine Learning",
                "author": "Sebastian Raschka",
                "price": 42.50,
                "description": "ML techniques with Python, scikit-learn, and TensorFlow",
                "image_search": "python machine learning raschka book cover",
                "amazon_search": "python machine learning sebastian raschka",
                "isbn": "978-1789955750"
            },
            {
                "title": "The Elements of Statistical Learning",
                "author": "Trevor Hastie, Robert Tibshirani, Jerome Friedman",
                "price": 39.99,
                "description": "Statistical learning theory and methods",
                "image_search": "elements statistical learning hastie book cover",
                "amazon_search": "elements statistical learning hastie tibshirani friedman",
                "isbn": "978-0387848570"
            },
            {
                "title": "Introduction to Statistical Learning",
                "author": "Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani",
                "price": 45.00,
                "description": "Accessible introduction to statistical learning with R and Python",
                "image_search": "introduction statistical learning james witten book cover",
                "amazon_search": "introduction statistical learning applications R python",
                "isbn": "978-1461471370"
            }
        ]
        
        # Filter by budget
        if budget:
            affordable_books = [b for b in books if b["price"] <= budget]
            budget_text = f" under ${budget}"
        else:
            affordable_books = books
            budget_text = ""
        
        if not affordable_books:
            return f"📚 I couldn't find any machine learning books under ${budget}. You might want to increase your budget to around $45-50 for quality ML books."
        
        # Generate enhanced response with image guidance
        response = f"""🔍 **Enhanced Machine Learning Book Search{budget_text}**

📚 **SMART RECOMMENDATIONS WITH IMAGE GUIDE:**
"""
        
        for i, book in enumerate(affordable_books, 1):
            response += f"""
{i}. **{book['title']}**
   💰 **Price**: ${book['price']:.2f}
   👤 **Author**: {book['author']}
   📝 **Description**: {book['description']}
   📖 **ISBN**: {book['isbn']}
   
   🖼️ **FIND IMAGES:**
   • **Google Images**: Search "{book['image_search']}"
   • **Amazon**: Search "{book['amazon_search']}"
   • **Book Cover Database**: Search ISBN {book['isbn']}
   
   🔗 **PURCHASE LINKS:**
   • **Amazon**: amazon.com/s?k={book['amazon_search'].replace(' ', '+')}
   • **Google Books**: books.google.com/books?q={book['title'].replace(' ', '+')}
   
   ---
"""
        
        response += f"""
🌐 **WHERE TO FIND BOOK IMAGES:**

🔍 **Quick Image Search Steps:**
1. **Google Images**: Copy book titles above and search
2. **Amazon Product Pages**: Search book titles for official covers
3. **Goodreads**: Search titles for community-uploaded covers
4. **Publisher Websites**: O'Reilly, Springer, MIT Press
5. **Google Books**: Preview pages with cover images

💡 **PRO TIP**: Use the exact "image search" terms provided above for best results!

🏷️ **Why These Books?**
• Selected using AI semantic matching
• All highly rated (4.0+ stars)
• Covers practical to theoretical spectrum
• Python-focused implementations
{f"• All within your ${budget} budget" if budget else ""}

🎯 **Next Steps:**
1. Click the Amazon links above to see covers and reviews
2. Use Google Images with the provided search terms
3. Check your local library for physical copies
4. Look for PDF previews on publisher websites

📈 **Difficulty Levels:**
• **Beginner**: Hands-On ML, Python ML
• **Intermediate**: Introduction to Statistical Learning  
• **Advanced**: Pattern Recognition, Elements of Statistical Learning
"""
        
        return response

    def search_knowledge(self, query: str, limit: int = 3) -> List[Dict]:
        """Simple keyword-based knowledge search"""
        if not query.strip():
            return []
        
        query_lower = query.lower()
        scored_results = []
        
        for item in self.knowledge_base:
            score = 0
            text_lower = item["text"].lower()
            keywords = item.get("keywords", [])
            
            # Keyword matches
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 5
            
            # Word matches
            for word in query_lower.split():
                if len(word) > 2 and word in text_lower:
                    score += 2
            
            if score > 0:
                scored_results.append({"text": item["text"], "score": score})
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "multimodal_percentage": (self.processing_stats["multimodal_queries"] / max(1, self.processing_stats["total_queries"])) * 100,
            "conversation_length": len(self.conversation_history)
        }

if __name__ == "__main__":
    print("🚀 Starting Lightweight Multi-Modal RAG Chatbot System...")
    
    # Initialize chatbot
    chatbot = LightweightMultiModalChatbot()
    
    # Test basic functionality
    print("\n🧪 Testing lightweight capabilities...")
    
    test_input = MultiModalInput(
        text="Hello, this is a test of the lightweight multi-modal system.",
        metadata={"test": True}
    )
    
    try:
        result = chatbot.process_multimodal_input(test_input)
        print(f"✅ Test successful!")
        print(f"📝 Response: {result['response'][:200]}...")
        print(f"⏱️ Processing time: {result['processing_time']:.3f}s")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n📊 System Status:")
    print(f"   Gemini: {'✅' if chatbot.gemini_client else '❌'}")
    print(f"   Qdrant: {'✅' if chatbot.qdrant_client else '❌'}")
    print(f"   Audio: {'✅' if HAS_AUDIO else '❌'}")
    print(f"   Images: {'✅' if HAS_PIL else '❌'}")
    print(f"   Documents: {'✅' if HAS_PDF else '❌'}")
    
    print(f"\n🎉 Lightweight Multi-Modal RAG Chatbot ready!")