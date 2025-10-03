<div>

  Gemini Pipeline | [Traditional OCR Pipeline](./doc/README_TRADITIONAL.md) 

</div>

# Gemini Extractor

A modern, AI-powered invoice information extraction module using Google's Gemini API for structured data extraction from Vietnamese invoices.

## Overview

The Gemini Extractor is an alternative extraction pipeline that leverages large language models (LLMs) to perform intelligent, template-agnostic extraction of invoice data. Instead of relying on fixed rules or trained models for specific invoice layouts, it uses Gemini's vision and text understanding capabilities to extract structured information from diverse invoice formats.

## Key Features

- **Template-agnostic extraction**: Works across different invoice layouts without retraining
- **Vision + OCR integration**: Can process both image and text inputs
- **Structured output**: Returns clean JSON with validated fields
- **Configurable prompts**: Easy to customize extraction instructions via YAML configs
- **Batch processing**: Support for processing multiple invoices in a single run
- **Vietnamese language optimized**: Prompts and validation tailored for Vietnamese invoices

## Architecture

```
Invoice Image → OCR (optional) → Gemini API → Structured JSON
```

The extractor can work in two modes:
1. **Direct vision mode**: Send images directly to Gemini's vision API
2. **OCR + text mode**: Use local OCR (PaddleOCR/custom models) then send extracted text to Gemini

## Installation

### Prerequisites

- Python 3.9+
- A Google Cloud account with Gemini API access
- API key for Google Generative AI

### Setup

1. Install required dependencies:

```bash
pip install google-generativeai pyyaml pillow
```

2. Set up your Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or add it to your config file:

```yaml
# config/gemini_config.yaml
api_key: "your-api-key-here"
model: "gemini-flash-latest"
```

## Configuration

The extractor uses YAML configuration files to control behavior. Example config:

```yaml
# config/gemini_config.yaml
api:
  key: ${GEMINI_API_KEY}  # or direct string
  model: "gemini-flash-latest"
  temperature: 0.1
  max_output_tokens: 2048

extraction:
  mode: "vision"  # or "text"
  prompt_template: "prompts/extraction_vi.txt"
  output_format: "json"
  
fields:
  - SELLER
  - ADDRESS
  - TIMESTAMP
  - PRODUCTS
  - TOTAL_COST
  
products_schema:
  - PRODUCT
  - NUM
  - VALUE
```

### Configuration Options

- `api.key`: Your Gemini API key (can use environment variables)
- `api.model`: Gemini model to use (`gemini-1.5-flash`, `gemini-1.5-pro`, etc.)
- `api.temperature`: Controls randomness (0.0-1.0, lower = more deterministic)
- `extraction.mode`: `vision` for image input, `text` for OCR text input
- `extraction.prompt_template`: Path to custom prompt file
- `fields`: List of top-level fields to extract
- `products_schema`: Schema for product line items

## Usage

### Basic Usage (Single Image)

```python
from src.gemini_extractor import GeminiExtractor

# Initialize extractor
extractor = GeminiExtractor(config_path="config/gemini_config.yaml")

# Extract from image
result = extractor.extract_from_image("path/to/invoice.jpg")

print(result)
# {
#   "SELLER": "VinCommerce",
#   "ADDRESS": "...",
#   "TIMESTAMP": "...",
#   "PRODUCTS": [...],
#   "TOTAL_COST": "..."
# }
```

### Using the Example Scripts

We provide ready-to-use example scripts:

**Basic extraction (single image):**

```bash
python examples/basic_gemini_extraction.py
```

This will:
- Load a sample invoice image
- Extract structured data using Gemini
- Save results to `output/extracted_invoice.json`
- Print the extracted data

**Batch extraction (multiple images):**

```bash
python examples/batch_gemini_extraction.py
```

This will:
- Process all images in `uploads/` folder
- Extract data from each invoice
- Save individual JSON files to `output/batch_results/`
- Generate a summary report in `output/batch_results/batch_summary.json`

### Advanced Usage (Custom Prompts)

You can customize the extraction prompt to improve accuracy or extract additional fields:

```python
extractor = GeminiExtractor(
    config_path="config/gemini_config.yaml",
    custom_prompt="Extract seller, date, and all product items with prices..."
)

result = extractor.extract_from_image("invoice.jpg")
```

### Using with OCR Text

If you already have OCR results:

```python
# Extract from OCR text instead of image
result = extractor.extract_from_text(ocr_text)
```

### Batch Processing

```python
from pathlib import Path

images = list(Path("uploads/").glob("*.jpg"))
results = extractor.batch_extract(images, output_dir="output/batch_results")

# Results is a list of dicts with metadata
for item in results:
    print(f"{item['filename']}: {item['status']}")
    if item['status'] == 'success':
        print(item['data'])
```

## Prompt Engineering

The quality of extraction depends heavily on the prompt. The default prompt is in `prompts/extraction_vi.txt`. 

Example prompt structure:

```
Bạn là một hệ thống trích xuất thông tin hóa đơn chuyên nghiệp.

Từ hình ảnh/văn bản hóa đơn, hãy trích xuất các thông tin sau:

1. SELLER: Tên cửa hàng/công ty
2. ADDRESS: Địa chỉ
3. TIMESTAMP: Ngày giờ bán hàng
4. PRODUCTS: Danh sách sản phẩm (mỗi sản phẩm gồm tên, số lượng, giá trị)
5. TOTAL_COST: Tổng tiền

Trả về kết quả dưới dạng JSON hợp lệ theo định dạng:
{
  "SELLER": "...",
  "ADDRESS": "...",
  ...
}
```

Tips for better prompts:
- Be specific about output format (JSON structure)
- Provide examples of expected output
- Include edge case handling (missing fields, multiple formats)
- Use Vietnamese for Vietnamese invoices

## Output Format

Standard JSON output:

```json
{
  "SELLER": "VinCommerce",
  "ADDRESS": "TP. Cẩm Phả, Quảng Ninh",
  "TIMESTAMP": "ngày bán: 15/08/2020 09:47",
  "PRODUCTS": [
    {
      "PRODUCT": "dưa hấu không hạt 27.500/KG x 3,396 KG",
      "NUM": "1",
      "VALUE": "93.390"
    },
    {
      "PRODUCT": "cải thảo 24.900/KG x 1,704 KG",
      "NUM": "1",
      "VALUE": "42.430"
    }
  ],
  "TOTAL_COST": "331.142"
}
```

## API Reference

### GeminiExtractor

Main class for invoice extraction.

**Methods:**

- `__init__(config_path: str, custom_prompt: str = None)`
  - Initialize extractor with config file
  
- `extract_from_image(image_path: str) -> dict`
  - Extract data from invoice image
  - Returns: Dictionary with extracted fields
  
- `extract_from_text(text: str) -> dict`
  - Extract data from OCR text
  - Returns: Dictionary with extracted fields
  
- `batch_extract(image_paths: list, output_dir: str = None) -> list`
  - Process multiple invoices
  - Returns: List of results with metadata

### Configuration Schema

See `config/gemini_config.yaml` for full schema documentation.

## Performance & Costs

**API Costs** (as of October 2025):
- Gemini 1.5 Flash: ~$0.00035 per image (very affordable)
- Gemini 1.5 Pro: ~$0.0035 per image (higher accuracy)

**Processing Time**:
- ~1-3 seconds per invoice (depends on model and image size)
- Batch processing can be parallelized

**Accuracy**:
- Works well on most standard Vietnamese invoices
- May need prompt tuning for unusual layouts
- Handles handwritten text better than traditional OCR

## Troubleshooting

### Common Issues

**1. API Key Error**
```
Error: Invalid API key
```
Solution: Check that your `GEMINI_API_KEY` environment variable is set correctly.

**2. Rate Limiting**
```
Error: Resource exhausted (quota)
```
Solution: Add delays between requests or upgrade your API quota.

**3. Poor Extraction Quality**
Solution: 
- Try using `gemini-1.5-pro` instead of `flash`
- Adjust temperature (lower = more consistent)
- Refine your prompt in `prompts/extraction_vi.txt`

**4. Empty Results**
Solution:
- Check image quality (resolution, clarity)
- Verify the prompt matches your invoice format
- Try OCR + text mode instead of vision mode

### Debug Mode

Enable detailed logging:

```python
extractor = GeminiExtractor(config_path="config/gemini_config.yaml")
extractor.set_debug(True)  # Prints API requests/responses

result = extractor.extract_from_image("invoice.jpg")
```

## Comparison with Traditional OCR Pipeline

| Feature | Gemini Extractor | Traditional Pipeline |
|---------|-----------------|---------------------|
| Setup complexity | Low (API key only) | High (models, weights) |
| Training required | No | Yes |
| Template flexibility | High | Low |
| Cost | API costs | GPU costs |
| Offline capable | No | Yes |
| Accuracy (varied layouts) | High | Medium |

## Examples in This Repo

- `examples/basic_gemini_extraction.py` - Single image extraction demo
- `examples/batch_gemini_extraction.py` - Batch processing demo
- `config/gemini_config.yaml` - Full configuration example
- `config/gemini_config_simple.yaml` - Minimal config for quick start

## Further Reading

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Prompt Engineering Guide](https://ai.google.dev/docs/prompt_best_practices)
- See `architect/README_GEMINI.md` for design decisions and architecture details

## License

This module is part of the OCR-invoice project. See project root LICENSE.

---

**Questions or issues?** Check the main project README or open an issue on GitHub.
