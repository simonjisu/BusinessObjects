from pydantic import BaseModel, Field
from typing import Optional

class Description(BaseModel):
    output: dict[str, dict[str, dict[str, str]]] = Field(description='Description of each column for all tables in the database')

class SQLResponse(BaseModel):
    full_sql_query: str = Field(description='The full SQL query.')
    rationale: list[str] = Field(description='The step-by-step reasoning to generate the SQL query. ')

class BODescription(BaseModel):
    output: str = Field(description='The a clear and concise summary of the complex data model.')
    rationale: list[str] = Field(description='The step-by-step reasoning to generate the SQL query.')

class DatabaseModel(BaseModel):
    db_id: str
    db_schema: dict[str, dict[str, str]]
    col_explanation: dict[str, dict[str, str]]
    foreign_keys: list[str]
    primary_keys: list[str]

class QuestionSQL(BaseModel):
    question: str
    sql: str
    source_tables: list[str] = []

class SparcSample(BaseModel):
    sample_id: int = -1
    db_id: str
    interactions: list[QuestionSQL]
    final: QuestionSQL
    
class BusinessObject(BaseModel):
    obj_id: str
    obj_name: Optional[str] = None
    virtual_table: str
    description: str

class SpiderSample(BaseModel):
    sample_id: int|str = None
    db_id: str
    final: QuestionSQL
    bo: Optional[BusinessObject] = None

class BirdSample(BaseModel):
    sample_id: int|str = None
    db_id: str
    final: QuestionSQL
    evidence: str
    bo: Optional[BusinessObject] = None

