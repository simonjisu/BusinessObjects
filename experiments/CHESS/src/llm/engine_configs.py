from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from typing import Dict, Any

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_o1_apikey = os.environ.get("OPENAI_O1_KEY")

# init vertexai project
import vertexai
vertexai.init(project="crypto-isotope-366706", location="us-central1")

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-1.5-pro": {
        "constructor": ChatVertexAI,
        "params": {"model": "gemini-1.5-pro-002", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-flash": {
        "constructor": ChatVertexAI,
        "params": {"model": "gemini-1.5-flash-002", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gpt-3.5-turbo-0125": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    },
    "gpt-3.5-turbo-instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-instruct", "temperature": 0}
    },
    "gpt-4o": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o", "temperature": 0, "api_key": openai_o1_apikey,}
    },
    "gpt-4-0125-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-0125-preview", "temperature": 0}
    },
    "gpt-4-turbo": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-turbo", "temperature": 0}
    },
    "gpt-4o-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini", "temperature": 0, "api_key": openai_o1_apikey,}
    },
    "claude-3-opus-20240229": {
        "constructor": ChatAnthropic,
        "params": {"model": "claude-3-opus-20240229", "temperature": 0}
    },
    "finetuned_nl2sql": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "AI4DS/NL2SQL_DeepSeek_33B",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 400,
            "temperature": 0,
            "model_kwargs": {
                "stop": ["```\n", ";"]
            }
        }
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {
                "stop": [""]
            }
        }
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {
                "stop": [""]
            }
        }
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1/chat/completions",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {
                "stop": [""]
            }
        }
    }
}
