from typing import List, Dict, Any
import logging
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """vLLM-based model implementation for efficient inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = None
        self.sampling_params = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the vLLM model."""
        # Lazy import vllm only when the model is actually initialized
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            logger.error(f"Failed to import vllm: {e}")
            raise ImportError("vllm is required for VLLMModel. Please install it with: pip install vllm")
        
        logger.info(f"Loading model: {self.model_path}")
        
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
        
        # Add stop sequences if provided
        if "stop" in gen_config:
            sampling_args["stop"] = gen_config["stop"]
        
        self.sampling_params = SamplingParams(**sampling_args)
        
        logger.info(f"Model {self.name} loaded successfully")
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate SQL queries for given prompts."""
        if not self.llm:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Generating responses for {len(prompts)} prompts")
        
        # Add system-level instruction to force SQL-only output
        system_instruction = "You are a SQL query generator. Output only valid SQL queries with no explanations, comments, or markdown formatting."
        
        # Modify prompts to include strong instruction
        modified_prompts = []
        for prompt in prompts:
            modified_prompt = f"{system_instruction}\n\n{prompt}\n\nSQL:"
            modified_prompts.append(modified_prompt)
        
        outputs = self.llm.generate(modified_prompts, self.sampling_params)
        
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
        logger.info(f"Model {self.name} cleaned up") 