from pydantic import BaseModel

class InputSchema(BaseModel):
    folder_id: str
