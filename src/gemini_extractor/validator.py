import re
from datetime import datetime
from typing import Dict, List, Any, Union
import json

class DataValidator:
    """
    Validates extracted invoice data
    """
    
    def validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """Check JSON schema compliance"""
        required_keys = {"SELLER", "TIMESTAMP", "PRODUCTS", "TOTAL_COST"}
        if not isinstance(data, dict):
            return False
        if not required_keys.issubset(data.keys()):
            missing = required_keys - data.keys()
            print(f"❌ Missing required keys: {missing}")
            return False
        if not isinstance(data["PRODUCTS"], list):
            print("❌ 'PRODUCTS' must be a list")
            return False
        return True

    def validate_field_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure correct data types and attempt to fix if possible"""
        try:
            # Ensure SELLER is string
            if not isinstance(data.get("SELLER"), str):
                data["SELLER"] = str(data.get("SELLER", ""))
            
            # Normalize TOTAL_COST to float
            total = data.get("TOTAL_COST")
            if isinstance(total, str):
                data["TOTAL_COST"] = self._parse_number(total)
            elif isinstance(total, (int, float)):
                data["TOTAL_COST"] = float(total)
            else:
                raise ValueError("Invalid TOTAL_COST type")
            
            # Process each product
            for prod in data["PRODUCTS"]:
                if not isinstance(prod, dict):
                    raise ValueError("Each product must be a dict")
                # Ensure NUM is int
                if isinstance(prod.get("NUM"), str):
                    prod["NUM"] = int(re.sub(r"[^\d]", "", prod["NUM"]) or "1")
                elif isinstance(prod.get("NUM"), (int, float)):
                    prod["NUM"] = int(prod["NUM"])
                else:
                    prod["NUM"] = 1
                
                # Normalize VALUE to float
                val = prod.get("VALUE")
                if isinstance(val, str):
                    prod["VALUE"] = self._parse_number(val)
                elif isinstance(val, (int, float)):
                    prod["VALUE"] = float(val)
                else:
                    prod["VALUE"] = 0.0

        except (ValueError, TypeError) as e:
            print(f"❌ Type validation error: {e}")
            raise
        return data

    def _parse_number(self, value: Union[str, float, int]) -> float:
        """Helper to parse number from string with commas, dots, etc."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove everything except digits and one decimal point
            # Handle cases like "12.300", "12,300", "12300đ"
            cleaned = re.sub(r"[^\d.,]", "", value)
            # Replace comma as thousand separator (common in Vietnam)
            if ',' in cleaned and cleaned.rfind(',') > cleaned.rfind('.'):
                # Likely comma is decimal (e.g., European style) → but in VN, comma = thousand
                # We assume: if more than 3 digits after last comma → comma = thousand
                parts = cleaned.split(',')
                if len(parts[-1]) == 3 and len(cleaned) > 4:
                    cleaned = cleaned.replace(',', '')
                else:
                    # Treat comma as decimal
                    cleaned = cleaned.replace(',', '.')
            # Remove extra dots (keep only last one as decimal)
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0

    def validate_total(self, products: List[Dict], total: float) -> bool:
        """Cross-check product sum vs total (allow small rounding tolerance)"""
        calculated = sum(p.get("NUM", 0) * p.get("VALUE", 0.0) for p in products)
        tolerance = 100  # Allow ±100 VND due to rounding or discounts
        is_valid = abs(calculated - total) <= tolerance
        if not is_valid:
            print(f"⚠️ Total mismatch: calculated={calculated:,}, given={total:,}")
        return is_valid


class DataNormalizer:
    """
    Normalizes and cleans extracted data
    """
    
    def parse_timestamp(self, timestamp: str) -> datetime:
        """Convert various Vietnamese date formats to standard datetime"""
        if not isinstance(timestamp, str):
            raise ValueError("Timestamp must be string")
        
        # Common Vietnamese formats
        formats = [
            "%d/%m/%Y %H:%M",
            "%d-%m-%Y %H:%M",
            "%d/%m/%y %H:%M",
            "%d-%m-%y %H:%M",
            "%d/%m/%Y",
            "%d-%m-%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {timestamp}")

    def normalize_currency(self, value: Union[str, float, int]) -> float:
        """Handle different currency formats (VND)"""
        validator = DataValidator()
        return validator._parse_number(value)