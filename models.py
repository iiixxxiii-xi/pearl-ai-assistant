"""Pydantic 请求/响应模型"""
from pydantic import BaseModel, Field


class ReplyRequest(BaseModel):
    question: str = Field(..., min_length=1, description="客户需求简述")
    trait: str = Field(default="", description="客户性格（选填）：比较纠结/很纠结/爽快果断")
    knowledge_level: str = Field(default="", description="对珍珠了解程度：完全不懂/略懂一点/很懂行")
    budget_range: str = Field(default="", description="预算范围：1000以内/1000-3000/3000-5000/5000以上")
    usage: str = Field(default="", description="用途：自己戴/送人")
    quality: str = Field(default="", description="品质要求：好看就行/品质好一点/要最好的")
    user: str = Field(default="mom", pattern="^(mom|sister)$", description="谁在回复：mom 或 sister")


class ContentRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="想写的话题")


class FeedbackRequest(BaseModel):
    action: str = Field(..., pattern="^(copied|regenerated)$")
    question: str = Field(default="")
    reply: str = Field(default="")


class ReplyResponse(BaseModel):
    reply: str
    sources: list[str]


class ContentResponse(BaseModel):
    content: str
    sources: list[str]


class StatsResponse(BaseModel):
    total: int
    copied: int
    regenerated: int
    adoption_rate: float
