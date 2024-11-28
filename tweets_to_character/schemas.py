from pydantic import BaseModel

class InputSchema(BaseModel):
    input_dir: str
