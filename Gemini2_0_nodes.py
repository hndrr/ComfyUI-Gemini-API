import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from google import genai
from google.genai import types
import traceback

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Google-Gemini"
    
    def __init__(self):
        """Initialize the logging system and API key storage"""
        self.log_messages = []  # Global log message storage
        # Get the directory where the node is located
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        
        # Check google-genai version
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"Current google-genai version: {genai_version}")
            
            # Check PIL/Pillow version
            try:
                import PIL
                self.log(f"Current PIL/Pillow version: {PIL.__version__}")
            except Exception as e:
                self.log(f"Unable to check PIL/Pillow version: {str(e)}")
            
            # Check if the version meets the minimum requirements
            from packaging import version
            if version.parse(genai_version) < version.parse('1.5.0'):  
                self.log("Warning: google-genai version is too low, recommend upgrading to the latest version")
                self.log("Suggested command: pip install -q -U google-genai")
            
            # Check if PIL/Pillow version meets requirements
            try:
                if version.parse(PIL.__version__) < version.parse('9.5.0'):
                    self.log("Warning: PIL/Pillow version is too low, recommend upgrading to 9.5.0 or higher")
                    self.log("Suggested command: pip install -U Pillow>=9.5.0")
            except Exception:
                pass
        except Exception as e:
            self.log(f"Unable to check version information: {e}")
    
    def log(self, message):
        """Global logging function: record to the log list"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """Get API key, prioritize user input key"""
        # If the user entered a valid key, use and save it
        if user_input_key and len(user_input_key) > 10:
            self.log("Using user input API key")
            # Save to file
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("API key saved to node directory")
            except Exception as e:
                self.log(f"Failed to save API key: {e}")
            return user_input_key
            
        # If user did not input, try to read from file
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("Using saved API key")
                    return saved_key
            except Exception as e:
                self.log(f"Failed to read saved API key: {e}")
                
        # If none available, return empty string
        self.log("Warning: No valid API key provided")
        return ""
    
    def generate_empty_image(self, width, height):
        """Generate standard format empty RGB image tensor - ensure ComfyUI compatible format [B,H,W,C]"""
        # Create an image tensor that meets ComfyUI standards
        # ComfyUI expects [batch, height, width, channels] format!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"Created ComfyUI compatible empty image: shape={tensor.shape}, type={tensor.dtype}")
        return tensor
    
    def validate_and_fix_tensor(self, tensor, name="image"):
        """Validate and fix tensor format, ensure full compatibility with ComfyUI"""
        try:
            # Basic shape check
            if tensor is None:
                self.log(f"Warning: {name} is None")
                return None
                
            self.log(f"Validating {name}: shape={tensor.shape}, type={tensor.dtype}, device={tensor.device}")
            
            # Ensure correct shape: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"Error: {name} has incorrect shape: {tensor.shape}")
                return None
                
            if tensor.shape[1] != 3:
                self.log(f"Error: {name} channel count is not 3: {tensor.shape[1]}")
                return None
                
            # Ensure type is float32
            if tensor.dtype != torch.float32:
                self.log(f"Correcting {name} type: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)
                
            # Ensure memory is contiguous
            if not tensor.is_contiguous():
                self.log(f"Correcting {name} memory layout: making it contiguous")
                tensor = tensor.contiguous()
                
            # Ensure value range is between 0-1
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 0 or max_val > 1:
                self.log(f"Correcting {name} value range: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)
                
            return tensor
        except Exception as e:
            self.log(f"Error validating tensor: {e}")
            traceback.print_exc()
            return None
    
    def generate_image(self, prompt, api_key, model, width, height, temperature, seed=66666666, image=None):
        """Generate image - using simplified API key management"""
        response_text = ""
        
        # Reset log messages
        self.log_messages = []
        
        try:
            # Get API key
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "Error: No valid API key provided. Please enter an API key in the node or ensure a key has been saved."
                self.log(error_message)
                full_text = "## Error\n" + error_message + "\n\n## Instructions\n1. Enter your Google API key in the node\n2. The key will be automatically saved to the node directory, so you won't need to enter it next time"
                return (self.generate_empty_image(width, height), full_text)
            
            # Create client instance
            client = genai.Client(api_key=actual_api_key)
            
            # Handle seed value
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"Generated random seed value: {seed}")
            else:
                self.log(f"Using specified seed value: {seed}")
            
            # Build prompt with dimension information
            aspect_ratio = width / height
            if aspect_ratio > 1:
                orientation = "landscape (horizontal)"
            elif aspect_ratio < 1:
                orientation = "portrait (vertical)"
            else:
                orientation = "square"

            simple_prompt = f"Create a detailed image of: {prompt}. Generate the image in {orientation} orientation with exact dimensions of {width}x{height} pixels. Ensure the composition fits properly within these dimensions without stretching or distortion."
            
            # Configure generation parameters, using user-specified temperature
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # Log temperature setting
            self.log(f"Using temperature: {temperature}, seed value: {seed}")
            
            # Process reference image
            contents = []
            has_reference = False
            
            if image is not None:
                try:
                    # Ensure image format is correct
                    if len(image.shape) == 4 and image.shape[0] == 1:  # [1, H, W, 3] format
                        # Get first frame image
                        input_image = image[0].cpu().numpy()
                        
                        # Convert to PIL image
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)
                        
                        self.log(f"Reference image processed successfully, dimensions: {pil_image.width}x{pil_image.height}")
                        
                        # Process directly in memory, don't save to file
                        img_byte_arr = BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        image_bytes = img_byte_arr.read()
                        
                        # Add image part and text part
                        img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                        txt_part = {"text": simple_prompt + " Use this reference image as style guidance."}
                        
                        # Combine content (image first, text after)
                        contents = [img_part, txt_part]
                        has_reference = True
                        self.log("Reference image added to request")
                    else:
                        self.log(f"Reference image format incorrect: {image.shape}")
                        contents = simple_prompt
                except Exception as img_error:
                    self.log(f"Reference image processing error: {str(img_error)}")
                    contents = simple_prompt
            else:
                # No reference image, use text only
                contents = simple_prompt
            
            # Print request information
            self.log(f"Requesting Gemini API to generate image, seed value: {seed}, includes reference image: {has_reference}")
            
            # Call API
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                config=gen_config
            )
            
            # Process response
            self.log("API response received successfully, processing...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("No candidates in API response")
                # Combine logs and return value
                full_text = "\n".join(self.log_messages) + "\n\nAPI returned empty response"
                return (self.generate_empty_image(width, height), full_text)
            
            # Check if response contains image
            image_found = False
            
            # Iterate through response parts
            for part in response.candidates[0].content.parts:
                # Check if it's a text part
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self.log(f"API returned text: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                
                # Check if it's an image part
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self.log("Processing API returned data")
                    try:
                        # Get image data
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "unknown"
                        self.log(f"Image data type: {type(image_data)}, MIME type: {mime_type}, data length: {len(image_data) if image_data else 0}")
                        
                        # Log first 8 bytes for diagnosis
                        if image_data and len(image_data) > 8:
                            hex_prefix = ' '.join([f'{b:02x}' for b in image_data[:8]])
                            self.log(f"First 8 bytes of image data: {hex_prefix}")
                            
                            # Detect Base64 encoded PNG
                            if hex_prefix.startswith('69 56 42 4f 52'):
                                try:
                                    self.log("Detected Base64 encoded PNG, decoding...")
                                    base64_str = image_data.decode('utf-8', errors='ignore')
                                    image_data = base64.b64decode(base64_str)
                                    self.log(f"Base64 decoding successful, new data length: {len(image_data)}")
                                except Exception as e:
                                    self.log(f"Base64 decoding failed: {str(e)}")
                        
                        # BytesIO correct usage method - modified for more direct initialization
                        try:
                            # Initialize BytesIO directly with byte data, more concise and compatible
                            buffer = BytesIO(image_data)
                            
                            # Try to open image
                            pil_image = Image.open(buffer)
                            self.log(f"Successfully opened image: {pil_image.width}x{pil_image.height}, format: {pil_image.format}")
                            
                            # Ensure RGB mode
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # Resize
                            if pil_image.width != width or pil_image.height != height:
                                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # Convert to ComfyUI format
                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            
                            self.log(f"Image successfully converted to tensor, shape: {img_tensor.shape}")
                            
                            # Combine logs and API return text
                            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
                            return (img_tensor, full_text)
                        
                        except Exception as e:
                            self.log(f"Failed to open image with BytesIO: {str(e)}")
                            self.log("Unable to process image data, using default empty image")
                            img_tensor = self.generate_empty_image(width, height)
                    except Exception as e:
                        self.log(f"Image processing error: {e}")
                        traceback.print_exc()  # Add detailed error trace information
            
            # No image data found, but might have text
            if not image_found:
                self.log("No image data found in API response, returning text only")
                if not response_text:
                    response_text = "API did not return any image or text"
            
            # Combine logs and API return text
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
            return (self.generate_empty_image(width, height), full_text)
        
        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            self.log(f"Gemini image generation error: {str(e)}")
            
            # Combine logs and error information
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
            return (self.generate_empty_image(width, height), full_text)

# 複数画像対応のGeminiジェネレーター
class GeminiMultiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image_multi"
    CATEGORY = "Google-Gemini"
    
    def __init__(self):
        """Initialize the logging system and API key storage"""
        self.log_messages = []  # Global log message storage
        # Get the directory where the node is located
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        
        # Check google-genai version
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"Current google-genai version: {genai_version}")
            
            # Check PIL/Pillow version
            try:
                import PIL
                self.log(f"Current PIL/Pillow version: {PIL.__version__}")
            except Exception as e:
                self.log(f"Unable to check PIL/Pillow version: {str(e)}")
            
            # Check if the version meets the minimum requirements
            from packaging import version
            if version.parse(genai_version) < version.parse('1.5.0'):  
                self.log("Warning: google-genai version is too low, recommend upgrading to the latest version")
                self.log("Suggested command: pip install -q -U google-genai")
            
            # Check if PIL/Pillow version meets requirements
            try:
                if version.parse(PIL.__version__) < version.parse('9.5.0'):
                    self.log("Warning: PIL/Pillow version is too low, recommend upgrading to 9.5.0 or higher")
                    self.log("Suggested command: pip install -U Pillow>=9.5.0")
            except Exception:
                pass
        except Exception as e:
            self.log(f"Unable to check version information: {e}")
    
    def log(self, message):
        """Global logging function: record to the log list"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """Get API key, prioritize user input key"""
        # If the user entered a valid key, use and save it
        if user_input_key and len(user_input_key) > 10:
            self.log("Using user input API key")
            # Save to file
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("API key saved to node directory")
            except Exception as e:
                self.log(f"Failed to save API key: {e}")
            return user_input_key
            
        # If user did not input, try to read from file
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("Using saved API key")
                    return saved_key
            except Exception as e:
                self.log(f"Failed to read saved API key: {e}")
                
        # If none available, return empty string
        self.log("Warning: No valid API key provided")
        return ""
    
    def generate_empty_image(self, width, height):
        """Generate standard format empty RGB image tensor - ensure ComfyUI compatible format [B,H,W,C]"""
        # Create an image tensor that meets ComfyUI standards
        # ComfyUI expects [batch, height, width, channels] format!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"Created ComfyUI compatible empty image: shape={tensor.shape}, type={tensor.dtype}")
        return tensor
    
    def process_image(self, image):
        """Process a single ComfyUI image into bytes for API request"""
        if image is None:
            return None
            
        try:
            # Ensure image format is correct
            if len(image.shape) == 4 and image.shape[0] == 1:  # [1, H, W, 3] format
                # Get first frame image
                input_image = image[0].cpu().numpy()
                
                # Convert to PIL image
                input_image = (input_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(input_image)
                
                # Process directly in memory
                img_byte_arr = BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes = img_byte_arr.read()
                
                self.log(f"Image processed successfully, dimensions: {pil_image.width}x{pil_image.height}")
                return {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
            else:
                self.log(f"Image format incorrect: {image.shape}")
                return None
        except Exception as e:
            self.log(f"Image processing error: {str(e)}")
            return None
    
    def generate_image_multi(self, prompt, api_key, model, width, height, temperature, seed=66666666, 
                             image1=None, image2=None, image3=None, image4=None):
        """Generate image with multiple reference images"""
        response_text = ""
        
        # Reset log messages
        self.log_messages = []
        
        try:
            # Get API key
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "Error: No valid API key provided. Please enter an API key in the node or ensure a key has been saved."
                self.log(error_message)
                full_text = "## Error\n" + error_message + "\n\n## Instructions\n1. Enter your Google API key in the node\n2. The key will be automatically saved to the node directory, so you won't need to enter it next time"
                return (self.generate_empty_image(width, height), full_text)
            
            # Create client instance
            client = genai.Client(api_key=actual_api_key)
            
            # Handle seed value
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"Generated random seed value: {seed}")
            else:
                self.log(f"Using specified seed value: {seed}")
            
            # Build prompt with dimension information
            aspect_ratio = width / height
            if aspect_ratio > 1:
                orientation = "landscape (horizontal)"
            elif aspect_ratio < 1:
                orientation = "portrait (vertical)"
            else:
                orientation = "square"

            simple_prompt = f"Create a detailed image of: {prompt}. Generate the image in {orientation} orientation with exact dimensions of {width}x{height} pixels. Ensure the composition fits properly within these dimensions without stretching or distortion."
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # Log temperature setting
            self.log(f"Using temperature: {temperature}, seed value: {seed}")
            
            # Process reference images
            contents = []
            image_parts = []
            image_count = 0
            
            # Process each image input if provided
            for idx, img in enumerate([image1, image2, image3, image4]):
                if img is not None:
                    processed = self.process_image(img)
                    if processed:
                        image_parts.append(processed)
                        image_count += 1
                        self.log(f"Reference image {idx+1} added to request")
            
            # Add text part with simple prompt
            txt_part = {"text": simple_prompt}
            if image_count > 0:
                txt_part["text"] += f" Use these {image_count} reference images as style/content guidance."
            
            # Combine all parts (images first, then text)
            contents = image_parts + [txt_part]
            
            # Print request information
            self.log(f"Requesting Gemini API to generate image with {image_count} reference images, seed value: {seed}")
            
            # Call API
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                config=gen_config
            )
            
            # Process response
            self.log("API response received successfully, processing...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("No candidates in API response")
                # Combine logs and return value
                full_text = "\n".join(self.log_messages) + "\n\nAPI returned empty response"
                return (self.generate_empty_image(width, height), full_text)
            
            # Check if response contains image
            image_found = False
            
            # Iterate through response parts
            for part in response.candidates[0].content.parts:
                # Check if it's a text part
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self.log(f"API returned text: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                
                # Check if it's an image part
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self.log("Processing API returned data")
                    image_found = True
                    try:
                        # Get image data
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "unknown"
                        self.log(f"Image data type: {type(image_data)}, MIME type: {mime_type}, data length: {len(image_data) if image_data else 0}")
                        
                        # Log first 8 bytes for diagnosis
                        if image_data and len(image_data) > 8:
                            hex_prefix = ' '.join([f'{b:02x}' for b in image_data[:8]])
                            self.log(f"First 8 bytes of image data: {hex_prefix}")
                            
                            # Detect Base64 encoded PNG
                            if hex_prefix.startswith('69 56 42 4f 52'):
                                try:
                                    self.log("Detected Base64 encoded PNG, decoding...")
                                    base64_str = image_data.decode('utf-8', errors='ignore')
                                    image_data = base64.b64decode(base64_str)
                                    self.log(f"Base64 decoding successful, new data length: {len(image_data)}")
                                except Exception as e:
                                    self.log(f"Base64 decoding failed: {str(e)}")
                        
                        # BytesIO correct usage method
                        try:
                            # Initialize BytesIO directly with byte data
                            buffer = BytesIO(image_data)
                            
                            # Try to open image
                            pil_image = Image.open(buffer)
                            self.log(f"Successfully opened image: {pil_image.width}x{pil_image.height}, format: {pil_image.format}")
                            
                            # Ensure RGB mode
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # Resize
                            if pil_image.width != width or pil_image.height != height:
                                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # Convert to ComfyUI format
                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            
                            self.log(f"Image successfully converted to tensor, shape: {img_tensor.shape}")
                            
                            # Combine logs and API return text
                            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
                            return (img_tensor, full_text)
                        
                        except Exception as e:
                            self.log(f"Failed to open image with BytesIO: {str(e)}")
                            self.log("Unable to process image data, using default empty image")
                            
                    except Exception as e:
                        self.log(f"Image processing error: {e}")
                        traceback.print_exc()
            
            # No image data found, but might have text
            if not image_found:
                self.log("No image data found in API response, returning text only")
                if not response_text:
                    response_text = "API did not return any image or text"
            
            # Combine logs and API return text
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
            return (self.generate_empty_image(width, height), full_text)
        
        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            self.log(f"Gemini multi-image generation error: {str(e)}")
            
            # Combine logs and error information
            full_text = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
            return (self.generate_empty_image(width, height), full_text)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "Google-Gemini": GeminiImageGenerator,
    "Google-Gemini-Multi": GeminiMultiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google-Gemini": "Gemini 2.0 image",
    "Google-Gemini-Multi": "Gemini 2.0 multi-image"
} 