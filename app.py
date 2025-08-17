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
            .info-btn {
                min-width: 40px !important;
                height: 40px !important;
                padding: 8px !important;
                border: 1px solid #e5e7eb !important;
                background: #f9fafb !important;
                border-radius: 6px !important;
            }
            .info-btn:hover {
                background: #f3f4f6 !important;
                border-color: #d1d5db !important;
            }
            .close-btn {
                min-width: 32px !important;
                height: 32px !important;
                padding: 4px !important;
                border: 1px solid #e5e7eb !important;
                background: #ffffff !important;
                border-radius: 4px !important;
            }
            .close-btn:hover {
                background: #f9fafb !important;
                border-color: #d1d5db !important;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # <img src="static/icons/solar--text-field-bold.svg" width="32" height="32" style="display: inline; vertical-align: middle; margin-right: 8px;"> LLM Text Classifier
                
                A high-performance microservice for text classification using Google's Gemini API.
                Supports multiple classification tasks with configurable prompts.
                """,
                elem_id="header"
            )
            
            with gr.Tabs():
                
                # Single Text Classification Tab
                with gr.Tab("Single Text Classification", icon="static/icons/fluent--slide-text-28-filled.svg"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                text_input = gr.Textbox(
                                    label="Text to Classify",
                                    placeholder="Enter the text you want to classify...",
                                    lines=5,
                                    max_lines=10,
                                    scale=4
                                )
                                with gr.Column(scale=1, min_width=50):
                                    info_button = gr.Button(
                                        "", 
                                        icon="static/icons/lets-icons--info-alt-duotone.svg",
                                        size="sm", 
                                        variant="secondary"
                                    )
                            
                            # Usage instructions (initially hidden)
                            with gr.Row(visible=False) as usage_info:
                                with gr.Column():
                                    with gr.Row():
                                        gr.Markdown("## Usage Instructions")
                                        close_info_button = gr.Button(
                                            "", 
                                            icon="static/icons/lets-icons--close-round-duotone.svg",
                                            size="sm", 
                                            variant="secondary"
                                        )
                                    
                                    gr.Markdown(
                                        """
### 1. Single Text Classification Tab (what you're seeing)

"Enter the text you want to classify" means you should type or paste any text that you want the AI to analyze. Here are some examples:

#### For Sentiment Analysis (Movie Reviews):
• **Positive example**: "This movie was absolutely fantastic! Great acting and amazing plot."
• **Negative example**: "Terrible movie, waste of time and money. Poor acting."

#### For Rating Classification (Product Reviews):
• **5-star example**: "Excellent product! Works perfectly, highly recommend!"
• **1-star example**: "Broke after one day, terrible quality."
• **3-star example**: "Average product, nothing special but does the job."

### 2. How to Use It:

1. Type your text in the text box (any review, comment, or text)
2. Select a task from the dropdown:
   • `sentiment` = Classify as positive/negative (like movie reviews)
   • `rating` = Predict star rating 1-5 (like product reviews)
3. Click "🚀 Classify"
4. See the result in JSON format below

### 3. What You'll Get:
The app will show you:
• **Prediction**: "positive"/"negative" or "1"/"2"/"3"/"4"/"5"
• **Response Time**: How long it took
• **Model**: Which AI model was used (gemini-2.0-flash)

### 4. Try These Examples:

**Copy and paste these into the text box:**

```
This smartphone is amazing! Battery lasts all day and camera quality is superb.
```
*(Try with `rating` task - should predict 4 or 5)*

```
Worst purchase ever. Product arrived broken and customer service was unhelpful.
```
*(Try with `rating` task - should predict 1 or 2)*

```
I loved this movie! The storyline was captivating and the actors did a brilliant job.
```
*(Try with `sentiment` task - should predict "positive")*

The AI is analyzing your text and using prompt engineering to classify it based on the patterns it learned during training!
                                        """
                                    )
                            
                            task_dropdown = gr.Dropdown(
                                choices=task_names,
                                label="Classification Task",
                                value=task_names[0] if task_names else None
                            )
                            
                            classify_btn = gr.Button(
                                "Classify Text", 
                                icon="static/icons/pepicons-print--play-circle-filled.svg",
                                variant="primary"
                            )
                        
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
                            
                            batch_classify_btn = gr.Button(
                                "Classify Batch", 
                                icon="static/icons/pepicons-print--play-circle-filled.svg",
                                variant="primary"
                            )
                        
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
            def show_usage_info():
                return gr.update(visible=True)
            
            def hide_usage_info():
                return gr.update(visible=False)
            
            info_button.click(
                fn=show_usage_info,
                outputs=[usage_info]
            )
            
            close_info_button.click(
                fn=hide_usage_info,
                outputs=[usage_info]
            )
            
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
