import os
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


class Configuration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> Self:
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: Dict[str, Any] = {
            f: os.environ.get(f.upper(), configurable.get(f))
            for f in cls.__pydantic_fields__.keys()
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
