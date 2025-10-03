#!/usr/bin/env python3
"""
Basic example of using the Gemini Invoice Extractor pipeline

This script demonstrates:
1. Single invoice extraction
2. Displaying results
3. Saving to file
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_extractor.pipeline import GeminiInvoiceExtractor
from gemini_extractor.config import load_config

# Load environment variables
load_dotenv()
config = load_config()

def main():
    """Main extraction example"""
    
    # Configuration
    API_KEY = os.getenv("GEMINI_API_KEY")
    IMAGE_PATH = "/home/anlab/Downloads/test-invoice/Image (2).jpeg"
    
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables")
        print("Please set it in your .env file or export it:")
        print("export GEMINI_API_KEY='your-api-key'")
        return
    
    # Initialize extractor
    print("Initializing Gemini Invoice Extractor...")
    extractor = GeminiInvoiceExtractor(config)
    
    # Extract and display
    print(f"\nProcessing invoice: {IMAGE_PATH}")
    print("=" * 60)
    
    try:
        result = extractor.extract_and_display(IMAGE_PATH)
        
        # Save to file
        output_path = Path("output") / "extracted_invoice.json"
        extractor.save_result(result, output_path, format="json")
        
        print("\nExtraction completed successfully!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\nExtraction failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    
    main()
