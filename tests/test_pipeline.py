#!/usr/bin/env python3
"""
Test script for Gemini Invoice Extractor Pipeline

Run this to verify your pipeline is working correctly.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_extractor.pipeline import GeminiInvoiceExtractor

def test_initialization():
    """Test 1: Can we initialize the extractor?"""
    print("Test 1: Initialization")
    print("-" * 40)
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ FAILED: GEMINI_API_KEY not found")
        print("Please set GEMINI_API_KEY in your .env file")
        return False
    
    try:
        extractor = GeminiInvoiceExtractor(
            api_key=api_key,
            system_prompt_path="prompts/extraction_vi.txt"
        )
        print("✅ PASSED: Extractor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def test_preprocessing():
    """Test 2: Can we preprocess an image?"""
    print("\nTest 2: Image Preprocessing")
    print("-" * 40)
    
    load_dotenv()
    
    try:
        extractor = GeminiInvoiceExtractor(
            api_key=os.getenv("GEMINI_API_KEY"),
            system_prompt_path="prompts/extraction_vi.txt"
        )
        
        # Check if test image exists
        test_image = "/home/anlab/Downloads/Image (2).jpeg"
        if not Path(test_image).exists():
            print(f"⚠️ SKIPPED: Test image not found: {test_image}")
            return True
        
        # Try preprocessing
        image = extractor._preprocess_image(test_image)
        print(f"✅ PASSED: Image preprocessed successfully")
        print(f"   Image size: {image.size}")
        print(f"   Image mode: {image.mode}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_extraction():
    """Test 3: Can we extract data from an invoice?"""
    print("\nTest 3: Full Extraction")
    print("-" * 40)
    
    load_dotenv()
    
    try:
        extractor = GeminiInvoiceExtractor(
            api_key=os.getenv("GEMINI_API_KEY"),
            system_prompt_path="prompts/extraction_vi.txt"
        )
        
        # Check if test image exists
        test_image = "/home/anlab/Downloads/Image (2).jpeg"
        if not Path(test_image).exists():
            print(f"⚠️ SKIPPED: Test image not found: {test_image}")
            return True
        
        print("Processing invoice (this may take a few seconds)...")
        result = extractor.extract(test_image)
        
        print("✅ PASSED: Extraction successful")
        print(f"   Seller: {result.get('SELLER', 'N/A')}")
        print(f"   Total Cost: {result.get('TOTAL_COST', 'N/A')}")
        print(f"   Products: {len(result.get('PRODUCTS', []))}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_validation():
    """Test 4: Does validation work?"""
    print("\nTest 4: Data Validation")
    print("-" * 40)
    
    try:
        from gemini_extractor.validator import DataValidator
        
        validator = DataValidator()
        
        # Test valid data
        valid_data = {
            "SELLER": "Test Store",
            "TIMESTAMP": "15/08/2020 09:47",
            "PRODUCTS": [
                {"PRODUCT": "Item 1", "NUM": "1", "VALUE": "100"}
            ],
            "TOTAL_COST": "100"
        }
        
        if validator.validate_json_structure(valid_data):
            print("✅ PASSED: Structure validation works")
        else:
            print("❌ FAILED: Structure validation failed")
            return False
        
        # Test type conversion
        normalized = validator.validate_field_types(valid_data)
        if isinstance(normalized['TOTAL_COST'], float):
            print("✅ PASSED: Type conversion works")
            return True
        else:
            print("❌ FAILED: Type conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 GEMINI INVOICE EXTRACTOR - PIPELINE TESTS")
    print("=" * 60)
    
    tests = [
        test_initialization,
        test_preprocessing,
        test_validation,
        test_extraction,  # This one hits the API
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test crashed: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
