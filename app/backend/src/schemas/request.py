from pydantic import BaseModel
from typing import List, Dict, Any

class KeypointsFrame(BaseModel):
    type: str
    results: Dict[str, Any]

class FullSequence(BaseModel):
    type: str
    sequence: List[KeypointsFrame]
    topic_id: str = None  # Thêm topic_id để chỉ định model nào sẽ được dùng

class WordItem(BaseModel):
    id: int
    label: str
    video_path: str
    topic_id: str = None  # Thêm topic_id cho mỗi word