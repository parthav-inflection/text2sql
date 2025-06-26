from typing import List, Dict, Any
import logging
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """vLLM-based model implementation for tool calling and summarization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the vLLM model."""
        # Lazy import vllm only when the model is actually initialized
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise ImportError("vllm and transformers are required for VLLMModel. Please install them with: pip install vllm transformers")
        
        logger.info(f"Loading model: {self.model_path}")
        
        # Load tokenizer for chat template support
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            logger.info(f"Tokenizer loaded for {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # vLLM configuration
        vllm_config = self.config.get("vllm_config", {})
        
        # Build vLLM initialization arguments
        llm_args = {
            "model": self.model_path,
            "tensor_parallel_size": vllm_config.get("tensor_parallel_size", 1),
            "max_model_len": vllm_config.get("max_model_len", 8192),
            "trust_remote_code": vllm_config.get("trust_remote_code", True)
        }
        
        # CPU-specific configuration
        if vllm_config.get("device") == "cpu":
            logger.info("Configuring for CPU inference")
            llm_args.update({
                "device": "cpu",
                "enforce_eager": vllm_config.get("enforce_eager", True),
                "disable_log_stats": vllm_config.get("disable_log_stats", True),
                "block_size": vllm_config.get("block_size", 16),
                "swap_space": vllm_config.get("swap_space", 4)
            })
        else:
            # GPU configuration
            llm_args["gpu_memory_utilization"] = vllm_config.get("gpu_memory_utilization", 0.9)
        
        self.llm = LLM(**llm_args)
        
        # Sampling parameters
        gen_config = self.config.get("generation_config", {})
        sampling_args = {
            "temperature": gen_config.get("temperature", 0.0),
            "max_tokens": gen_config.get("max_tokens", 512),
            "top_p": gen_config.get("top_p", 1.0),
            "top_k": gen_config.get("top_k", -1)
        }
        
        # Add stop sequences and stop_token_ids automatically
        self._configure_stop_tokens(sampling_args)
        
        self.sampling_params = SamplingParams(**sampling_args)
        
        logger.info(f"Model {self.name} loaded successfully")
    
    def _configure_stop_tokens(self, sampling_args: Dict[str, Any]):
        """Automatically configure stop tokens from tokenizer."""
        # Start with default stop sequences
        stop_sequences = ["</s>", "<|end|>", "<|endoftext|>"]
        stop_token_ids = []
        
        if self.tokenizer:
            try:
                # Get EOS token ID
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    stop_token_ids.append(self.tokenizer.eos_token_id)
                
                # For Qwen-based models (including OmniSQL), add the im_end token
                if self._is_chat_model():
                    try:
                        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                        if im_end_id is not None and im_end_id not in stop_token_ids:
                            stop_token_ids.append(im_end_id)
                            logger.debug(f"Added Qwen chat stop token <|im_end|> (id: {im_end_id})")
                    except:
                        pass
                
                logger.info(f"Configured stop tokens: {stop_token_ids}")
                
            except Exception as e:
                logger.warning(f"Could not configure stop tokens: {e}")
                stop_token_ids = [2]  # Common EOS fallback
        
        sampling_args["stop"] = stop_sequences
        if stop_token_ids:
            sampling_args["stop_token_ids"] = stop_token_ids
    
    def _is_chat_model(self) -> bool:
        """Detect if this is a chat model that needs chat template formatting."""
        chat_indicators = [
            "OmniSQL", "Instruct", "Chat", "chat", "-it", "instruct",
            "alpaca", "vicuna", "wizard", "orca", "XiYan", "Qwen"
        ]
        
        model_name_lower = self.model_path.lower()
        has_chat_indicator = any(indicator.lower() in model_name_lower for indicator in chat_indicators)
        
        # Also check if tokenizer has chat template
        has_chat_template = (
            self.tokenizer and 
            hasattr(self.tokenizer, 'chat_template') and 
            self.tokenizer.chat_template is not None
        )
        
        return has_chat_indicator or has_chat_template
    
    def _format_prompt_for_model(self, prompt: str) -> str:
        """Format prompt using chat template if available."""
        if self._is_chat_model() and self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Create a chat message
                messages = [{"role": "user", "content": prompt}]
                
                # Apply chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                logger.debug(f"Applied chat template formatting")
                return formatted_prompt
                
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, falling back to raw prompt")
                return prompt
        else:
            # For base models, add minimal instruction wrapper
            return f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"

    def generate(self, prompts: List[str], temperature_override: float = None) -> List[str]:
        """Generate responses for given prompts with optional temperature override."""
        if not self.llm:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Generating responses for {len(prompts)} prompts")
        
        # Format prompts based on model type
        formatted_prompts = [self._format_prompt_for_model(prompt) for prompt in prompts]
        
        # Use custom sampling params if temperature override is provided
        sampling_params = self.sampling_params
        if temperature_override is not None:
            from vllm import SamplingParams
            logger.info(f"Using temperature override: {temperature_override}")
            
            # Create new sampling params with overridden temperature
            override_args = {
                "temperature": temperature_override,
                "max_tokens": self.sampling_params.max_tokens,
                "top_p": self.sampling_params.top_p,
                "top_k": self.sampling_params.top_k,
                "stop": self.sampling_params.stop,
            }
            if hasattr(self.sampling_params, 'stop_token_ids'):
                override_args["stop_token_ids"] = self.sampling_params.stop_token_ids
                
            sampling_params = SamplingParams(**override_args)
        
        # Generate responses
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # Extract text from outputs
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)
        
        return results
    
    def cleanup(self):
        """Clean up model resources."""
        if self.llm:
            del self.llm
            self.llm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        logger.info(f"Model {self.name} cleaned up") 