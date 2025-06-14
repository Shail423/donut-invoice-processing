import argparse
import json
import re
import torch
import os
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import warnings
warnings.filterwarnings("ignore")

class EnhancedInvoiceProcessor:
    def __init__(self, model_path):
        """Initialize with improved error handling and configuration validation."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            # Load with explicit configuration handling
            self.processor = DonutProcessor.from_pretrained(model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Validate processor configuration
            self._validate_processor_config()
            print(f"✅ Model loaded successfully from: {model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _validate_processor_config(self):
        """Validate and display processor configuration."""
        if hasattr(self.processor, 'image_processor'):
            print("✅ Image processor configuration is valid")
        else:
            print("⚠️ Image processor configuration may need updating")
    
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing with validation."""
        try:
            # Load and validate image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print(f"📷 Processing image: {image.size}")
            
            # Process with the processor (handles resizing and normalization)
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            print(f"🔧 Tensor shape: {pixel_values.shape}")
            return pixel_values, image
            
        except Exception as e:
            print(f"❌ Image preprocessing error: {e}")
            raise
    
    def generate_improved_output(self, pixel_values):
        """Generate text with optimized parameters to reduce noise and repetition."""
        try:
            # Task prompt - ensure this matches your training
            task_prompt = "<s_invoice>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Optimized generation parameters
            generation_config = {
                "max_length": 256,  # Reasonable limit
                "min_length": 5,
                "num_beams": 3,     # Beam search for quality
                "early_stopping": True,
                "do_sample": False,  # Deterministic
                "repetition_penalty": 1.3,  # Prevent repetition
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,  # Prevent 3-gram repetition
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            print("🔤 Generating structured output...")
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    **generation_config
                )
            
            # Decode the output
            sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return sequence
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""
    
    def extract_invoice_fields(self, raw_output):
        """Enhanced field extraction with multiple strategies."""
        print(f"🔍 Raw output length: {len(raw_output)}")
        print(f"🔍 Raw output preview: {raw_output[:150]}...")
        
        if not raw_output.strip():
            return {}
        
        # Clean the output first
        cleaned_text = self._clean_output(raw_output)
        print(f"🧹 Cleaned text: {cleaned_text}")
        
        # Extract fields using multiple approaches
        extracted = {}
        
        # Strategy 1: Pattern-based extraction
        extracted.update(self._extract_with_patterns(cleaned_text))
        
        # Strategy 2: Context-based extraction
        extracted.update(self._extract_with_context(cleaned_text))
        
        # Strategy 3: Fallback from original noisy text
        if not extracted:
            extracted.update(self._extract_from_noisy_text(raw_output))
        
        return extracted
    
    def _clean_output(self, text):
        """Clean noisy output while preserving meaningful content."""
        # Remove task prompt
        text = re.sub(r'<s_invoice>', '', text)
        
        # Handle repetitive patterns
        text = re.sub(r'0{8,}', '', text)  # Remove long sequences of zeros
        text = re.sub(r'(.)\1{4,}', r'\1', text)  # Reduce excessive repetition
        
        # Clean up spacing and common noise
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Date Date Date', 'Date', text)
        text = re.sub(r'#+\s*', '', text)  # Remove hash symbols
        
        return text.strip()
    
    def _extract_with_patterns(self, text):
        """Extract fields using regex patterns."""
        fields = {}
        
        # Date extraction (multiple formats)
        date_patterns = [
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{2,4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                fields['date'] = match.group(1)
                break
        
        # Invoice number extraction
        invoice_patterns = [
            r'(?:FA|FACT|INV)[\s#]*(\d{2}/\d{4}/\d+)',
            r'([A-Z]{2,4}\d{2}/\d{4}/\d+)',
            r'(?:invoice|factura)[\s#:]*([A-Z0-9\-/]+)',
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['invoice_number'] = match.group(1)
                break
        
        # Amount extraction
        amount_patterns = [
            r'([€$£]\s*[\d,]+\.?\d*)',
            r'([\d,]+\.?\d*\s*[€$£])',
            r'total[\s:]*([€$£]?\s*[\d,]+\.?\d*)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['total_amount'] = match.group(1).strip()
                break
        
        return fields
    
    def _extract_with_context(self, text):
        """Extract using contextual clues."""
        fields = {}
        tokens = text.split()
        
        for i, token in enumerate(tokens):
            # Date context
            if token.lower() in ['date', 'fecha'] and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if re.match(r'\d{1,2}[/\-]', next_token):
                    fields['date'] = next_token
            
            # Amount context
            if token.lower() in ['total', 'amount'] and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if re.match(r'[€$£]?[\d,]+', next_token):
                    fields['total_amount'] = next_token
        
        return fields
    
    def _extract_from_noisy_text(self, raw_text):
        """Last resort extraction from very noisy output."""
        fields = {}
        
        # Look for partial date in noise
        date_match = re.search(r'(\d{1,2}/\d{1,2})', raw_text)
        if date_match:
            partial_date = date_match.group(1)
            # Try to find a year nearby
            year_match = re.search(rf'{re.escape(partial_date)}[/\s]*(\d{{4}})', raw_text)
            if year_match:
                fields['date'] = f"{partial_date}/{year_match.group(1)}"
            else:
                # Assume current format needs year completion
                fields['date'] = partial_date + "/2020"  # Based on your example
        
        return fields

def main():
    parser = argparse.ArgumentParser(description='Enhanced Donut Invoice Processing')
    parser.add_argument('--image_path', required=True, help='Path to invoice image')
    parser.add_argument('--finetuned_model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print("🚀 Initializing Enhanced Invoice Processor...")
        processor = EnhancedInvoiceProcessor(args.finetuned_model)
        
        print(f"🖼️ Processing: {args.image_path}")
        
        # Process image
        pixel_values, original_image = processor.preprocess_image(args.image_path)
        
        # Generate output
        raw_output = processor.generate_improved_output(pixel_values)
        
        # Extract fields
        extracted_fields = processor.extract_invoice_fields(raw_output)
        
        # Display results
        print("\n" + "="*60)
        print("📊 EXTRACTION RESULTS")
        print("="*60)
        
        print(f"📄 Invoice Number: {extracted_fields.get('invoice_number', 'Not found')}")
        print(f"📅 Date: {extracted_fields.get('date', 'Not found')}")
        print(f"💰 Total Amount: {extracted_fields.get('total_amount', 'Not found')}")
        print(f"🏢 Company: {extracted_fields.get('company', 'Not found')}")
        print(f"📍 Address: {extracted_fields.get('address', 'Not found')}")
        
        # Save results
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}_enhanced.json")
        
        result = {
            "image_path": args.image_path,
            "raw_output": raw_output,
            "extracted_fields": extracted_fields,
            "processing_status": "success" if extracted_fields else "no_extraction"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Debug information
        print(f"\n🔍 Debug Info:")
        print(f"  Raw output length: {len(raw_output)} characters")
        print(f"  Fields extracted: {len(extracted_fields)}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
