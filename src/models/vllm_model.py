from typing import List, Dict, Any
import logging
from vllm import LLM, SamplingParams
from .base import BaseModel

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
        logger.info(f"Loading model: {self.model_path}")
        
        # vLLM configuration
        vllm_config = self.config.get("vllm_config", {})
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
            max_model_len=vllm_config.get("max_model_len", 8192),
            trust_remote_code=vllm_config.get("trust_remote_code", True)
        )
        
        # Sampling parameters
        gen_config = self.config.get("generation_config", {})
        self.sampling_params = SamplingParams(
            temperature=gen_config.get("temperature", 0.0),
            max_tokens=gen_config.get("max_tokens", 512),
            top_p=gen_config.get("top_p", 1.0),
            top_k=gen_config.get("top_k", -1)
        )
        
        logger.info(f"Model {self.name} loaded successfully")
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate SQL queries for given prompts."""
        if not self.llm:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Generating responses for {len(prompts)} prompts")
        outputs = self.llm.generate(prompts, self.sampling_params)
        
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