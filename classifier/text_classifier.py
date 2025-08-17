import logging
import time
from typing import Dict, Any, List, Optional
from .config_manager import ConfigManager
from .gemini_service import GeminiService

logger = logging.getLogger(__name__)

class TextClassifier:
    """Main text classifier that orchestrates the classification process."""
    
    def __init__(self, config_path: str = "config/classifier_config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize the LLM service
        self.llm_service = GeminiService(self.config)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60 / self.config['api']['rate_limit']  # seconds between requests
        
        logger.info("TextClassifier initialized successfully")
    
    def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def classify(self, text: str, task: str) -> Dict[str, Any]:
        """
        Classify text using the specified task.
        
        Args:
            text: Text to classify
            task: Classification task name
            
        Returns:
            Dictionary containing classification results and metadata
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if task not in self.config_manager.get_available_tasks():
            raise ValueError(f"Unknown task: {task}. Available tasks: {self.config_manager.get_available_tasks()}")
        
        # Apply rate limiting
        self._rate_limit()
        
        # Get task configuration
        task_config = self.config_manager.get_task_config(task)
        
        try:
            # Perform classification
            result = self.llm_service.classify_text(text, task_config)
            
            # Add task metadata
            result.update({
                'task_name': task,
                'task_type': task_config.get('type', 'classification'),
                'input_text': text[:200] + "..." if len(text) > 200 else text,
                'timestamp': time.time()
            })
            
            logger.info(f"Classification completed for task '{task}': {result['prediction']}")
            return result
            
        except Exception as e:
            logger.error(f"Classification failed for task '{task}': {str(e)}")
            raise
    
    def batch_classify(self, texts: List[str], task: str) -> List[Dict[str, Any]]:
        """
        Classify multiple texts using the specified task.
        
        Args:
            texts: List of texts to classify
            task: Classification task name
            
        Returns:
            List of classification results
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        results = []
        total_texts = len(texts)
        
        logger.info(f"Starting batch classification of {total_texts} texts for task '{task}'")
        
        for i, text in enumerate(texts, 1):
            try:
                result = self.classify(text, task)
                result['batch_index'] = i - 1
                results.append(result)
                
                if i % 10 == 0:  # Log progress every 10 items
                    logger.info(f"Processed {i}/{total_texts} texts")
                    
            except Exception as e:
                logger.error(f"Failed to classify text {i}/{total_texts}: {str(e)}")
                # Add error result
                results.append({
                    'batch_index': i - 1,
                    'error': str(e),
                    'input_text': text[:200] + "..." if len(text) > 200 else text,
                    'timestamp': time.time()
                })
        
        logger.info(f"Batch classification completed: {len(results)} results")
        return results
    
    def get_available_tasks(self) -> List[Dict[str, Any]]:
        """Get information about available classification tasks."""
        tasks = []
        for task_name in self.config_manager.get_available_tasks():
            task_config = self.config_manager.get_task_config(task_name)
            tasks.append({
                'name': task_name,
                'display_name': task_config['name'],
                'description': task_config['description'],
                'type': task_config.get('type', 'classification'),
                'classes': task_config['classes']
            })
        return tasks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return self.llm_service.get_model_info()
    
    def reload_config(self) -> None:
        """Reload configuration and reinitialize services."""
        logger.info("Reloading configuration...")
        self.config_manager.reload_config()
        self.config = self.config_manager.get_config()
        self.llm_service = GeminiService(self.config)
        logger.info("Configuration reloaded successfully")
