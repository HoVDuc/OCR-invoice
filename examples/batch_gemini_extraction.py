#!/usr/bin/env python3
"""
Batch processing example for Gemini Invoice Extractor

This script demonstrates:
1. Processing multiple invoices from a directory
2. Saving all results
3. Generating summary statistics
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_extractor.pipeline import GeminiInvoiceExtractor, BatchProcessor
from gemini_extractor.config import load_config

# Load environment variables
load_dotenv()
config = load_config()

def main():
    """Batch processing example"""
    
    # Configuration
    INPUT_DIR = "/home/anlab/Downloads/test-invoice"  # Directory with invoice images
    OUTPUT_DIR = "output/batch_results"
    
    if not config.get('api.api_key'):
        print("Error: GEMINI_API_KEY not found in environment variables")
        return
    
    # Check if input directory exists
    if not Path(INPUT_DIR).exists():
        print(f"Input directory not found: {INPUT_DIR}")
        print("Creating sample directory...")
        Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Please add invoice images to {INPUT_DIR} and run again")
        return
    
    # Initialize extractor
    print("Initializing Gemini Invoice Extractor...")
    extractor = GeminiInvoiceExtractor(config)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(extractor)
    
    # Process all invoices
    print(f"Processing invoices from: {INPUT_DIR}")
    print(f"Saving results to: {OUTPUT_DIR}")
    print("=" * 60)
    
    results = batch_processor.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        continue_on_error=True
    )
    
    # Display summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total invoices: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success']/max(results['total'], 1)*100:.1f}%")
    
    # Show errors if any
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {Path(error['image']).name}: {error['error']}")
    
    # Save summary
    summary_path = Path(OUTPUT_DIR) / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nSummary saved to: {summary_path}")
    print("Batch processing complete!")

if __name__ == "__main__":
    main()
