"""
Gemini Invoice Extraction Pipeline
Main orchestrator that coordinates all components for invoice extraction
"""
import os
import json
import traceback
from typing import Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
from loguru import logger

from .preprocessor import ImageProcessor
from .gemini_client import GeminiClient
from .validator import DataValidator, DataNormalizer
from .formatter import to_table


class GeminiInvoiceExtractor:
    """
    Main pipeline for extracting invoice information using Gemini-Flash VLM
    
    This class orchestrates the entire extraction process:
    1. Image preprocessing
    2. Prompt building  
    3. Gemini API call
    4. Response parsing
    5. Data validation & normalization
    6. Output formatting
    """
    
    def __init__(self, exp):
        """
        Initialize the Gemini Invoice Extractor
        
        Args:
            api_key: Google Gemini API key
            system_prompt_path: Path to system prompt file
            model_version: Gemini model version to use
            max_image_size: Maximum image dimension (will resize if larger)
            validate_strict: If True, raise errors on validation failures
            auto_normalize: If True, automatically normalize extracted data
        """
        # Initialize components
        self.preprocessor = ImageProcessor()
        self.gemini_client = GeminiClient(exp)
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()
        
        # Configuration
        self.max_image_size = exp.get('preprocessing.max_image_size', 1024)
        self.validate_strict = exp.get('validation.strict_mode', True)
        self.auto_normalize = exp.get('normalization.auto_normalize', True)

        logger.info(f"GeminiInvoiceExtractor initialized with model: {self.gemini_client.model_version}")

    def extract(
        self,
        image_path: Union[str, Path],
        prompt: str = "Extract invoice data to JSON",
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Extract invoice information from image
        
        Args:
            image_path: Path to invoice image file
            prompt: User prompt for extraction (default uses system prompt)
            return_raw: If True, return raw response without validation
            
        Returns:
            Dictionary containing extracted invoice data
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If validation fails in strict mode
            Exception: If API call or parsing fails
        """
        image_path = str(image_path)
        logger.info(f"Starting extraction for: {image_path}")
        
        try:
            # Step 1: Validate and preprocess image
            processed_image = self._preprocess_image(image_path)
            
            # Step 2: Call Gemini API
            response = self._call_gemini_api(processed_image, prompt)
            
            # Step 3: Parse JSON from response
            data_dict = self._parse_response(response)
            
            if return_raw:
                return data_dict
            
            # Step 4: Validate and normalize
            validated_data = self._validate_and_normalize(data_dict)
            
            logger.info(f"Extraction successful for: {image_path}")
            return validated_data
            
        except Exception as e:
            logger.error(f"Extraction failed for {image_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        Validate and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object, preprocessed and ready for API
        """
        logger.info("Preprocessing image...")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate image format and size
        if not self.preprocessor.validate_image(image_path):
            raise ValueError(f"Invalid image file: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed (Gemini requires RGB)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Resize if image is too large
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_image_size:
            logger.info(f"Resizing image from {width}x{height} to fit {self.max_image_size}px")
            if height > width:
                image = self.preprocessor.resize_to_height(image, self.max_image_size)
            else:
                # Resize based on width
                scale = self.max_image_size / width
                new_size = (self.max_image_size, int(height * scale))
                image = self.preprocessor.resize_image(image, new_size)
        
        logger.info(f"Image preprocessed: {image.size}")
        return image
    
    def _call_gemini_api(self, image: Image.Image, prompt: str) -> Any:
        """
        Call Gemini API with image and prompt
        
        Args:
            image: PIL Image object
            prompt: User prompt
            
        Returns:
            Gemini API response object
        """
        logger.info("Calling Gemini API...")
        
        try:
            response = self.gemini_client.generate_content(
                image=image,
                prompt=prompt
            )
            logger.info("Received response from Gemini API")
            return response
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """
        Parse JSON from Gemini response
        
        Args:
            response: Gemini API response object
            
        Returns:
            Parsed dictionary
        """
        logger.info("Parsing response...")
        
        try:
            # Extract text from response
            text = response.text
            
            # Remove markdown code blocks if present
            text = text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            data_dict = json.loads(text)
            logger.info("Successfully parsed JSON response")
            return data_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.error(f"Raw response text: {text[:500]}...")
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except AttributeError as e:
            logger.error(f"Response has no text attribute: {str(e)}")
            raise ValueError("Invalid response format from Gemini API")
    
    def _validate_and_normalize(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize extracted data
        
        Args:
            data_dict: Raw extracted data
            
        Returns:
            Validated and normalized data
        """
        logger.info("✔️  Validating and normalizing data...")
        
        try:
            # Step 1: Validate JSON structure
            if not self.validator.validate_json_structure(data_dict):
                msg = "Invalid JSON structure"
                if self.validate_strict:
                    raise ValueError(msg)
                logger.warning(f"{msg}, continuing with partial data")
            
            # Step 2: Validate and fix field types
            data_dict = self.validator.validate_field_types(data_dict)
            
            if self.auto_normalize:
                # Step 3: Normalize seller & timestamp
                data_dict["SELLER"] = data_dict.get("SELLER", "").strip()
                
                try:
                    timestamp_str = data_dict.get("TIMESTAMP", "")
                    # if timestamp_str:
                    #     data_dict["TIMESTAMP"] = self.normalizer.parse_timestamp(timestamp_str)
                except ValueError as e:
                    logger.warning(f"Timestamp parsing error: {e}")
                    data_dict["TIMESTAMP"] = timestamp_str  # Keep as string
                
                # Step 4: Normalize currency values
                for prod in data_dict.get("PRODUCTS", []):
                    prod["VALUE"] = self.normalizer.normalize_currency(prod.get("VALUE", 0))
                
                # Normalize total cost
                data_dict["TOTAL_COST"] = self.normalizer.normalize_currency(
                    data_dict.get("TOTAL_COST", 0)
                )
            
            # Step 5: Validate totals (optional warning)
            products = data_dict.get("PRODUCTS", [])
            total = data_dict.get("TOTAL_COST", 0)
            if products and total:
                self.validator.validate_total(products, total)
            
            logger.info("Data validated and normalized")
            return data_dict
            
        except Exception as e:
            logger.error(f"Validation/normalization failed: {str(e)}")
            if self.validate_strict:
                raise
            logger.warning("Returning data with validation errors")
            return data_dict
    
    def extract_and_display(self, image_path: Union[str, Path], prompt: str = "Extract invoice data to JSON") -> Dict[str, Any]:
        """
        Extract and display results in a formatted table
        
        Args:
            image_path: Path to invoice image
            prompt: Extraction prompt
            
        Returns:
            Extracted and validated data
        """
        result = self.extract(image_path, prompt)
        
        # Display formatted output
        try:
            to_table(result)
        except Exception as e:
            logger.warning(f"Failed to display table: {e}")
            # Fallback to JSON output
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
    
    def save_result(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save extraction result to file
        
        Args:
            result: Extracted data dictionary
            output_path: Path to save file
            format: Output format ('json', 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved result to: {output_path}")
        
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"SELLER: {result.get('SELLER', '')}\n")
                f.write(f"TIMESTAMP: {result.get('TIMESTAMP', '')}\n")
                f.write(f"TOTAL_COST: {result.get('TOTAL_COST', '')}\n\n")
                f.write("PRODUCTS:\n")
                for i, prod in enumerate(result.get('PRODUCTS', []), 1):
                    f.write(f"  {i}. {prod.get('PRODUCT', '')}\n")
                    f.write(f"     Quantity: {prod.get('NUM', '')}\n")
                    f.write(f"     Value: {prod.get('VALUE', '')}\n")
            logger.info(f"Saved result to: {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")


class BatchProcessor:
    """
    Process multiple invoices in batch
    """
    
    def __init__(self, extractor: GeminiInvoiceExtractor):
        """
        Initialize batch processor
        
        Args:
            extractor: GeminiInvoiceExtractor instance
        """
        self.extractor = extractor
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.{jpg,jpeg,png}",
        continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing invoice images
            output_dir: Directory to save results (optional)
            pattern: Glob pattern for image files
            continue_on_error: If True, continue processing on errors
            
        Returns:
            Dictionary with processing statistics and results
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(input_dir.glob(f"*.{ext}"))
            image_files.extend(input_dir.glob(f"*.{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'extractions': [],
            'errors': []
        }
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                # Extract
                data = self.extractor.extract(image_path)
                
                # Save if output directory specified
                if output_dir:
                    output_file = output_dir / f"{image_path.stem}.json"
                    self.extractor.save_result(data, output_file)
                
                results['extractions'].append({
                    'image': str(image_path),
                    'success': True,
                    'data': data
                })
                results['success'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {str(e)}")
                results['extractions'].append({
                    'image': str(image_path),
                    'success': False,
                    'error': str(e)
                })
                results['failed'] += 1
                results['errors'].append({
                    'image': str(image_path),
                    'error': str(e)
                })
                
                if not continue_on_error:
                    raise
        
        logger.info(f"Batch processing complete: {results['success']}/{results['total']} successful")
        return results


# Convenience function for quick usage
def extract_invoice(
    image_path: str,
    api_key: str,
    system_prompt_path: str,
    display: bool = True
) -> Dict[str, Any]:
    """
    Quick extraction function for single invoice
    
    Args:
        image_path: Path to invoice image
        api_key: Gemini API key
        system_prompt_path: Path to system prompt file
        display: If True, display formatted table
        
    Returns:
        Extracted invoice data
    """
    extractor = GeminiInvoiceExtractor(
        api_key=api_key,
        system_prompt_path=system_prompt_path
    )
    
    if display:
        return extractor.extract_and_display(image_path)
    else:
        return extractor.extract(image_path)