"""
PROJECT DROWNED SIGNAL - FIXED VERSION
Mission: Process and analyze DeepSeek AI model outputs with robust error handling
and state management.

Architectural Decisions:
1. Use Firebase Firestore for state persistence (ecosystem requirement)
2. Implement retry logic with exponential backoff for transient failures
3. Structured logging for ecosystem observability
4. Configuration via environment variables for deployment flexibility
5. Type hints for maintainability and error prevention
"""

import os
import sys
import logging
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

# Third-party imports with error handling
try:
    import requests
    from requests.exceptions import RequestException, Timeout
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1 import Client as FirestoreClient
    from google.cloud.exceptions import GoogleCloudError
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Install with: pip install requests firebase-admin google-cloud-firestore")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('drowned_signal.log')
    ]
)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enum for tracking processing state in Firestore"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class SignalConfig:
    """Configuration for signal processing"""
    deepseek_api_endpoint: str = "https://api.deepseek.com/v1/chat/completions"
    max_retries: int = 3
    retry_delay_base: float = 2.0  # Base delay for exponential backoff
    timeout_seconds: int = 30
    batch_size: int = 10
    firestore_collection: str = "drowned_signals"
    
    @classmethod
    def from_env(cls) -> 'SignalConfig':
        """Create config from environment variables with defaults"""
        return cls(
            deepseek_api_endpoint=os.getenv('DEEPSEEK_API_ENDPOINT', 
                                          'https://api.deepseek.com/v1/chat/completions'),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            retry_delay_base=float(os.getenv('RETRY_DELAY_BASE', '2.0')),
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', '30')),
            batch_size=int(os.getenv('BATCH_SIZE', '10')),
            firestore_collection=os.getenv('FIRESTORE_COLLECTION', 'drowned_signals')
        )


class DeepSeekClient:
    """Client for DeepSeek API with robust error handling"""
    
    def __init__(self, api_key: str, config: SignalConfig):
        self.api_key = api_key
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("DeepSeekClient initialized")
    
    def call_model_with_retry(self, 
                            prompt: str, 
                            model: str = "deepseek-chat",
                            context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Call DeepSeek model with exponential backoff retry logic
        
        Args:
            prompt: Input prompt for the model
            model: Model identifier
            context: Additional context for the request
            
        Returns:
            Tuple of (response_text, metadata)
        """
        metadata = {
            'attempts': 0,
            'errors': [],
            'start_time': datetime.now(timezone.utc).isoformat()
        }
        
        for attempt in range(self.config.max_retries):
            metadata['attempts'] = attempt + 1
            current_delay = self.config.