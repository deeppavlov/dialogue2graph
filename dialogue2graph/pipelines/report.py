import json
import pandas as pd
from typing import List
from pydantic import BaseModel, Field, ConfigDict
from dialogue2graph import metrics
from pathlib import Path


class PipelineReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service: str = Field(description="Name of the service")
    properties: dict = Field(
        default_factory=dict,
        description="dictionary with essential report data like time, metrics etc",
    )
    subreports: List[metrics.DGReportType] = Field(
        default_factory=list, description="reports from pipeline steps"
    )

    def add_property(self, property_name: str, value):
        self.properties[property_name] = value

    def add_subreport(self, subreport: metrics.DGReportType):
        self.subreports.append(subreport)

    def to_json(self, path: Path = "report.json"):
        with open(path, "w") as f:
            json.dumps(self.model_dump_json(), f, indent=4)

    def to_csv(self):
        raise NotImplementedError(
            "CSV export is not implemented yet. Please use to_json() instead."
        )

    def to_html(self):
        raise NotImplementedError(
            "HTML export is not implemented yet. Please use to_json() instead."
        )

    def to_markdown(self):
        raise NotImplementedError(
            "Markdown export is not implemented yet. Please use to_json() instead."
        )

    def to_text(self, path: Path = "report.txt"):
        with open(path, "w") as f:
            f.write(str(self))

    def __str__(self):
        return f"PipelineReport(service={self.service}, properties={self.properties}, subreports={self.subreports})"
