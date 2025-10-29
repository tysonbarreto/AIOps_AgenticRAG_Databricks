from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, BaseModel
from typing import List
from dataclasses import dataclass
import os

class AIConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    openai_api_key:str
    llm_model:str = "gpt-5-nano-2025-08-07"
    chunk_size:int = 500
    chunk_overlap:int = 50
    default_urls:List[str]=[
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

@dataclass
class AIConfig:
    settings = AIConfigSettings()
    @property
    def activate_LLM_environment(self):
        """Initialize the LLM Model"""
        os.environ['OPENAI_API_KEY'] = self.settings.openai_api_key

    @property
    def llm_model(self):
        return self.settings.llm_model
        
if __name__=="__main__":
    __all__=["AIConfigSettings","AIConfig"]
        
        