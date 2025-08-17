import gradio as gr
import logging
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier.text_classifier import TextClassifier
from classifier.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassifierApp:
    """Gradio application for the LLM text classifier."""
    
    def __init__(self):
        try:
            self.classifier = TextClassifier()
            self.available_tasks = self.classifier.get_available_tasks()
            logger.info("Classifier app initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            raise
    
    def classify_single_text(self, text: str, task: str) -> Tuple[str, str]:
        """
        Classify a single text input.
        
        Returns:
            Tuple of (result_json, status_message)
        """
        if not text.strip():
            return "", "⚠️ Please enter some text to classify"
        
        if not task:
            return "", "⚠️ Please select a classification task"
        
        try:
            result = self.classifier.classify(text, task)
            
            # Format result for display
            formatted_result = {
                "Prediction": result['prediction'],
                "Task": result['task_name'],
                "Model": result['model'],
                "Response Time": f"{result['response_time']:.2f}s",
                "Timestamp": result['timestamp']
            }
            
            result_json = json.dumps(formatted_result, indent=2)
            status = f"✅ Classification completed: **{result['prediction']}**"
            
            return result_json, status
            
        except Exception as e:
            error_msg = f"❌ Classification failed: {str(e)}"
            logger.error(error_msg)
            return "", error_msg
    
    def classify_batch_text(self, file, task: str) -> Tuple[str, str]:
        """
        Classify text from uploaded file.
        
        Returns:
            Tuple of (results_csv, status_message)
        """
        if file is None:
            return "", "⚠️ Please upload a CSV file"
        
        if not task:
            return "", "⚠️ Please select a classification task"
        
        try:
            # Read the uploaded file
            df = pd.read_csv(file.name)
            
            # Determine text column (look for common names)
            text_columns = ['text', 'review', 'Text', 'Review', 'content', 'Content']
            text_column = None
            
            for col in text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                return "", f"⚠️ Could not find text column. Available columns: {list(df.columns)}"
            
            # Get texts to classify
            texts = df[text_column].astype(str).tolist()
            
            if len(texts) > 100:  # Limit for demo purposes
                texts = texts[:100]
                status_prefix = f"⚠️ Limited to first 100 rows. "
            else:
                status_prefix = ""
            
            # Perform batch classification
            results = self.classifier.batch_classify(texts, task)
            
            # Create results DataFrame
            results_df = pd.DataFrame([
                {
                    'text': r.get('input_text', ''),
                    'prediction': r.get('prediction', 'ERROR'),
                    'response_time': r.get('response_time', 0),
                    'error': r.get('error', '')
                }
                for r in results
            ])
            
            # Save to CSV
            output_path = "batch_results.csv"
            results_df.to_csv(output_path, index=False)
            
            successful = len([r for r in results if 'error' not in r])
            status = f"{status_prefix}✅ Batch classification completed: {successful}/{len(results)} successful"
            
            return output_path, status
            
        except Exception as e:
            error_msg = f"❌ Batch classification failed: {str(e)}"
            logger.error(error_msg)
            return "", error_msg
    
    def get_task_info(self, task: str) -> str:
        """Get information about the selected task."""
        if not task:
            return "Select a task to see details"
        
        try:
            task_details = next(t for t in self.available_tasks if t['name'] == task)
            
            info = f"""
**{task_details['display_name']}**

*Description:* {task_details['description']}

*Type:* {task_details['type']}

*Classes:* {', '.join(task_details['classes'])}
            """
            return info.strip()
        except:
            return "Task information not available"
    
    def get_model_info(self) -> str:
        """Get information about the current model."""
        try:
            model_info = self.classifier.get_model_info()
            
            info = f"""
**Model Information**

*Provider:* {model_info['provider']}
*Model:* {model_info['model_name']}
*Temperature:* {model_info['temperature']}
*Max Tokens:* {model_info['max_tokens']}
            """
            return info.strip()
        except:
            return "Model information not available"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        # Get task names for dropdown
        task_names = [task['name'] for task in self.available_tasks]
        
        with gr.Blocks(
            title="LLM Text Classifier",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .tab-nav {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 🤖 LLM Text Classifier
                
                A high-performance microservice for text classification using Google's Gemini API.
                Supports multiple classification tasks with configurable prompts.
                """,
                elem_id="header"
            )
            
            with gr.Tabs():
                
                # Single Text Classification Tab
                with gr.Tab("📝 Single Text Classification"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_input = gr.Textbox(
                                label="Text to Classify",
                                placeholder="Enter the text you want to classify...",
                                lines=5,
                                max_lines=10
                            )
                            
                            task_dropdown = gr.Dropdown(
                                choices=task_names,
                                label="Classification Task",
                                value=task_names[0] if task_names else None
                            )
                            
                            classify_btn = gr.Button("🚀 Classify", variant="primary")
                        
                        with gr.Column(scale=1):
                            task_info = gr.Markdown(
                                value=self.get_task_info(task_names[0] if task_names else ""),
                                label="Task Information"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            result_output = gr.Code(
                                label="Classification Result",
                                language="json"
                            )
                            
                            status_output = gr.Markdown(
                                value="Ready to classify text",
                                label="Status"
                            )
                
                # Batch Classification Tab
                with gr.Tab("📊 Batch Classification"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                label="Upload CSV File",
                                file_types=[".csv"],
                                type="filepath"
                            )
                            
                            batch_task_dropdown = gr.Dropdown(
                                choices=task_names,
                                label="Classification Task",
                                value=task_names[0] if task_names else None
                            )
                            
                            batch_classify_btn = gr.Button("🚀 Classify Batch", variant="primary")
                        
                        with gr.Column():
                            batch_info = gr.Markdown(
                                """
                                **Instructions:**
                                1. Upload a CSV file with text data
                                2. The file should have a column named 'text', 'review', 'Text', 'Review', 'content', or 'Content'
                                3. Select a classification task
                                4. Click 'Classify Batch' to process
                                
                                *Note: Limited to 100 rows for demo purposes*
                                """
                            )
                    
                    with gr.Row():
                        batch_result_file = gr.File(
                            label="Download Results",
                            visible=False
                        )
                        
                        batch_status = gr.Markdown(
                            value="Upload a CSV file to get started",
                            label="Status"
                        )
                
                # Model Information Tab
                with gr.Tab("ℹ️ Model Information"):
                    model_info_display = gr.Markdown(
                        value=self.get_model_info(),
                        label="Current Model Configuration"
                    )
                    
                    available_tasks_display = gr.Dataframe(
                        value=pd.DataFrame(self.available_tasks),
                        label="Available Classification Tasks",
                        interactive=False
                    )
            
            # Event handlers
            task_dropdown.change(
                fn=self.get_task_info,
                inputs=[task_dropdown],
                outputs=[task_info]
            )
            
            classify_btn.click(
                fn=self.classify_single_text,
                inputs=[text_input, task_dropdown],
                outputs=[result_output, status_output]
            )
            
            batch_classify_btn.click(
                fn=self.classify_batch_text,
                inputs=[file_input, batch_task_dropdown],
                outputs=[batch_result_file, batch_status]
            )
        
        return interface

def main():
    """Main function to run the Gradio app."""
    try:
        app = ClassifierApp()
        interface = app.create_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()
