import os
from PIL import Image

class ImageProcessor:
    
    def __init__(self):
        pass

    def validate_image(self, image_path) -> bool:
        # Check format
        is_valid = True
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if not any(image_path.lower().endswith(ext) for ext in valid_formats):
            print("Unsupported image format.")
            is_valid = False
        
        # Check if file can be opened
        try:
            with open(image_path, 'rb') as f:
                f.read()
        except Exception as e:
            print(f"Error reading image: {e}")
            is_valid = False

        # Check size (e.g., max 5MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if os.path.getsize(image_path) > max_size:
            print("Image file is too large.")
            is_valid = False

        return is_valid
    
    def resize_to_height(self, image, H) -> Image.Image:
        width, height = image.size

        # Calculate the scaling factor
        r = H/height
        return image.resize((int(width * r), H))

    def resize_image(self, image, size) -> Image.Image:
        return image.resize(size)

    def enhance_image(self, image) -> Image.Image:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2)
    
if __name__ == "__main__":
    processor = ImageProcessor()
    img_path = "/home/anlab/Downloads/Image.jpeg"
    
    if processor.validate_image(img_path):
        img = Image.open(img_path)
        img_resized = processor.resize_to_height(img, 1000)
        img_enhanced = processor.enhance_image(img_resized)
        img_enhanced.show()
    else:
        print("Invalid image file.")