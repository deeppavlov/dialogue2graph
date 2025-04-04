from typing import List
from pydantic import BaseModel, Field, ConfigDict
from dialogue2graph import metrics


class PipelineReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service: str = Field(description="Name of the service")
    properties: dict = Field(default_factory=dict, description="dictionary with essential report data like time, metrics etc")
    subreports: List[metrics.DGReportType] = Field(default_factory=list, description="reports from pipeline steps")

    def add_property(self, property_name: str, value):
        self.properties[property_name] = value

    def add_subreport(self, subreport: metrics.DGReportType):
        self.subreports.append(subreport)
