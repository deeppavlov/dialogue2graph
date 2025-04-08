import json
import pandas as pd
from typing import List
from pydantic import BaseModel, Field, ConfigDict
from dialogue2graph import metrics
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)


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
            json.dump(self.model_dump(), f, indent=4)

    def to_csv(self, path: Path = "report.csv"):
        """Export the report to a CSV file."""
        # Convert the properties and subreports to a DataFrame
        properties_df = pd.DataFrame.from_dict(self.properties, orient="index")
        properties_df.columns = ["Value"]
        properties_df.index.name = "Property"
        subreports_df = pd.DataFrame(
            [subreport for subreport in self.subreports]
        )
        subreports_df.index.name = "Subreport"
        # Concatenate the properties and subreports DataFrames
        report_df = pd.concat([properties_df, subreports_df], axis=1)
        # Save the DataFrame to a CSV file
        report_df.to_csv(path, index=True)

    def to_html(self):
        raise NotImplementedError(
            "HTML export is not implemented yet. Please use to_json() instead."
        )

    def to_markdown(self, path: Path = "report.md"):
        """Export the report to a Markdown file."""
        # Convert the properties and subreports to a Markdown string
        markdown_str = f"# Report for {self.service}\n\n"
        markdown_str += "## Metrics\n"
        for key, value in self.properties.items():
            if key == "time":
                markdown_str += f"- **{key}**: {value:.2f} seconds\n"
            elif isinstance(value, bool):
                markdown_str += f"- **{key}**: {'✅' if value else '❌'}\n"
            elif isinstance(value, dict):
                markdown_str += f"- **{key}**:\n"
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, bool):
                        markdown_str += f"  - **{sub_key}**: {'✅' if sub_value else '❌'}\n"
                    else:
                        markdown_str += f"  - **{sub_key}**: {sub_value}\n"
            else:
                markdown_str += f"- **{key}**: {value}\n"
        markdown_str += "\n## Subreports\n"
        for subreport in self.subreports:
            markdown_str += f"- **{subreport}**\n"
        
        # Save the Markdown string to a file
        with open(path, "w") as f:
            f.write(markdown_str)

    def to_text(self, path: Path = "report.txt"):
        with open(path, "w") as f:
            f.write(str(self))

    def __str__(self):
        """String representation of the report with color formatting."""
        header = f"{Fore.CYAN}Report for {self.service}{Style.RESET_ALL}"
        properties = f"{Fore.YELLOW}Metrics:{Style.RESET_ALL}"
        for key, value in self.properties.items():
            if key == "time":
                properties += f"\n  {key}: {Fore.GREEN}{value:.2f} seconds{Style.RESET_ALL}"
            elif isinstance(value, bool):
                color = Fore.GREEN if value else Fore.RED
                properties += f"\n  {key}: {color}{value}{Style.RESET_ALL}"
            elif isinstance(value, dict):
                properties += f"\n  {key}:"
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, bool):
                        color = Fore.GREEN if sub_value else Fore.RED
                        properties += f"\n    {Fore.MAGENTA}{sub_key}{Style.RESET_ALL}: {color}{sub_value}{Style.RESET_ALL}"
                    else:
                        properties += f"\n    {Fore.MAGENTA}{sub_key}{Style.RESET_ALL}: {sub_value}"
            else:
                properties += f"\n  {key}: {value}"
        subreports = f"{Fore.BLUE}Subreports:{Style.RESET_ALL}"
        for subreport in self.subreports:
            subreports += f"\n  {subreport}"
        return f"{header}\n{properties}\n{subreports}"
    
    # def __repr__(self):
    #     return f"PipelineReport(service={self.service}, properties={self.properties}, subreports={self.subreports})"
