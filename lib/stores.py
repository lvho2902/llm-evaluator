from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

@dataclass
class BaseModel:
    id: str
    name: str
    info: Optional[Dict] = None
    owned_by: str = 'ollama'  # 'ollama' or 'openai'

@dataclass
class OpenAIModel(BaseModel):
    owned_by: str = 'openai'
    external: bool = False
    source: Optional[str] = None

@dataclass
class OllamaModelDetails:
    parent_model: str
    format: str
    family: str
    families: Optional[List[str]] = None
    parameter_size: str
    quantization_level: str

@dataclass
class OllamaModel(BaseModel):
    owned_by: str = 'ollama'
    details: OllamaModelDetails
    size: int
    description: str
    model: str
    modified_at: str
    digest: str
    ollama: Optional[Dict] = None

@dataclass
class Settings:
    models: Optional[List[str]] = None
    conversationMode: Optional[bool] = None
    speechAutoSend: Optional[bool] = None
    responseAutoPlayback: Optional[bool] = None
    audio: Optional[Dict] = None
    showUsername: Optional[bool] = None
    notificationEnabled: Optional[bool] = None
    title: Optional[Dict] = None
    splitLargeDeltas: Optional[bool] = None
    chatDirection: str = 'LTR'  # 'LTR' or 'RTL'

    system: Optional[str] = None
    requestFormat: Optional[str] = None
    keepAlive: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[str] = None
    repeat_penalty: Optional[str] = None
    top_k: Optional[str] = None
    top_p: Optional[str] = None
    num_ctx: Optional[str] = None
    num_batch: Optional[str] = None
    num_keep: Optional[str] = None
    options: Optional[Dict] = None

@dataclass
class Prompt:
    command: str
    user_id: str
    title: str
    content: str
    timestamp: int

@dataclass
class Document:
    collection_name: str
    filename: str
    name: str
    title: str

@dataclass
class Config:
    status: bool
    name: str
    version: str
    default_locale: str
    default_models: str
    default_prompt_suggestions: List['PromptSuggestion']
    features: Dict[str, bool]
    oauth: Dict[str, Dict[str, str]]

@dataclass
class PromptSuggestion:
    content: str
    title: List[str]

@dataclass
class SessionUser:
    id: str
    email: str
    name: str
    role: str
    profile_image_url: str
