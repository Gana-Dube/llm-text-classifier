import os
import logging
import time
from typing import Dict, Any, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google's Gemini API for text classification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=config['classifier']['model_name']
        )
        
        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=config['classifier']['temperature'],
            max_output_tokens=config['classifier']['max_tokens']
        )
        
        logger.info(f"Initialized Gemini service with model: {config['classifier']['model_name']}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def classify_text(self, text: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify text using Gemini API.
        
        Args:
            text: Text to classify
            task_config: Configuration for the classification task
            
        Returns:
            Dictionary containing classification result and metadata
        """
        try:
            # Format the prompt
            prompt = task_config['prompt_template'].format(text=text)
            
            # Generate response
            start_time = time.time()
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            end_time = time.time()
            
            # Extract the classification result
            result_text = response.text.strip().lower()
            
            # Validate result against expected classes
            expected_classes = [cls.lower() for cls in task_config['classes']]
            
            if result_text not in expected_classes:
                # Try to find a match in the response
                for cls in expected_classes:
                    if cls in result_text:
                        result_text = cls
                        break
                else:
                    logger.warning(f"Unexpected classification result: {result_text}")
                    result_text = expected_classes[0]  # Default to first class
            
            return {
                'prediction': result_text,
                'confidence': 1.0,  # Gemini doesn't provide confidence scores
                'response_time': end_time - start_time,
                'model': self.config['classifier']['model_name'],
                'task': task_config['name']
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini classification: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'provider': 'gemini',
            'model_name': self.config['classifier']['model_name'],
            'temperature': self.config['classifier']['temperature'],
            'max_tokens': self.config['classifier']['max_tokens']
        }
