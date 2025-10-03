import random
import time
import google.generativeai as genai
from loguru import logger
from google.api_core import exceptions as google_exceptions


class GeminiClient:
    """
    Manages API authentication and requests
    """

    def __init__(self, exp):
        """Initialize with API credentials"""
        self.exp = exp
        self.model_version = exp.get('api.model_version', 'gemini-1.5-flash')
        genai.configure(api_key=exp.get('api.api_key'))
        generation_config = genai.GenerationConfig(
            temperature=exp.get('api.generation.temperature', 0.2),
            top_p=exp.get('api.generation.top_p', 0.8),
            top_k=exp.get('api.generation.top_k', 40),
            candidate_count=exp.get('api.generation.candidate_count', 1),
        )
        system_instruction = self.load_system_prompt(exp.get('prompts.system_prompt_path'))
        self.model = genai.GenerativeModel(self.model_version,
                                           system_instruction=system_instruction,
                                           generation_config=generation_config)
        
    def load_system_prompt(self, prompt_path: str) -> str:
        """Load system prompt from file"""
        with open(prompt_path, 'r') as file:
            return file.read()
        
    def generate_content(self, image: bytes, prompt: str) -> dict:
        """Send request to Gemini API"""
        logger.info("Sending request to Gemini API...")
        response = self.handle_rate_limits(self.model.generate_content, [prompt, image],
                                           max_retries=self.exp.get('api.retry.max_retries', 5),
                                           base_delay=self.exp.get('api.retry.base_delay', 1.0),
                                           max_delay=self.exp.get('api.retry.max_delay', 60.0))
        if not self.validate_response(response):
            logger.error("Invalid response from Gemini API")
            raise ValueError("Invalid response from Gemini API")
        return response
        
    def handle_rate_limits(self, func, *args, max_retries=5, base_delay=1.0, max_delay=60.0):
        """
        Implement retry logic with exponential backoff and jitter for rate-limited requests.
        
        Args:
            func: Callable function to retry (e.g., self.model.generate_content)
            *args: Arguments to pass to the function
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
        
        Returns:
            Result of the function if successful
        
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(max_retries + 1):
            try:
                return func(*args)
            except google_exceptions.ResourceExhausted as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    if attempt == max_retries:
                        logger.error(f"❌ Max retries reached for rate limit: {e}")
                        raise
                else:
                    # Not a rate limit error → re-raise immediately
                    raise
            except google_exceptions.ServiceUnavailable as e:
                # May also be retried (e.g., backend overload)
                if attempt == max_retries:
                    raise
            except google_exceptions.DeadlineExceeded as e:
                if attempt == max_retries:
                    raise

            # Calculate delay with exponential backoff + jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 0.1 * delay)
            total_delay = delay + jitter
            logger.warning(f"⏳ Rate limit or service error. Retrying in {total_delay:.2f}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(total_delay)

        raise RuntimeError("Unexpected state: should not reach here")

        
    def validate_response(self, response) -> bool:
        """
        Check if the Gemini API response is valid and contains usable content.
        
        Args:
            response: GenerateContentResponse from google.generativeai
            
        Returns:
            bool: True if response is valid and contains text, False otherwise
        """
        # Check response existence
        if not response:
            return False
        # Check candidates existence
        if not hasattr(response, 'candidates') or not response.candidates:
            logger.error("❌ No candidates in response")
            return False

        candidate = response.candidates[0]

        # Check finish reason
        finish_reason = str(candidate.finish_reason).upper()
        if finish_reason in ["SAFETY", "RECITATION", "BLOCKED", "OTHER"]:
            logger.error(f"❌ Response blocked or unsafe. Finish reason: {finish_reason}")
            return False
        elif finish_reason == "MAX_TOKENS":
            logger.warning("⚠️ Response may be incomplete (MAX_TOKENS), but will attempt to use partial content.")

        # Check if parts and text exist
        try:
            _ = response.text  # Triggers validation internally
            return True
        except (ValueError, AttributeError) as e:
            logger.error(f"❌ No valid text in response: {e}")
            return False
