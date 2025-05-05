""" Module for CLI entrypoints
"""
import click
from dotenv import load_dotenv
from .commands.generate_data import generate_data
from .commands.generate_graph_light import generate_light
from .commands.generate_graph_llm import generate_llm
from .commands.generate_graph_extender import generate_extender


@click.group()
def cli():
    """Dialog2Graph CLI tool for generating dialog graphs and data"""
    pass


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--topic", "-t", help="Topic for dialog generation", required=True)
@click.option("--output", "-o", help="Output file path", default="generated_data.json")
def gen_data(env: str, cfg: str, topic: str, output: str):
    """Generate dialog data for a given topic"""
    load_dotenv(env)
    generate_data(topic, cfg, output)


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogs", "-d", help="Input dialogs file", required=True)
@click.option("--tgraph", "-tg", help="Input true graph file", required=False)
@click.option("--output", "-o", help="Output graph file", required=False)
@click.option("--report", "-r", help="Output report file", required=False)
@click.option("--eval", "-ev", is_flag=True, help="Call pipeline evals", required=False)
def gen_graph_light(
    env: str, cfg: str, dialogs: str, tgraph: str, output: str, report: str, eval: bool
):
    """Generate graph from dialogs data via d2g_algo pipeline"""
    load_dotenv(env)
    generate_light(
        dialogs=dialogs,
        tgraph=tgraph,
        enable_evals=eval,
        config=cfg,
        graph_path=output,
        report_path=report,
    )


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogs", "-d", help="Input dialogs file", required=True)
@click.option("--tgraph", "-tg", help="Input true graph file", required=False)
@click.option("--output", "-o", help="Output graph file", required=False)
@click.option("--report", "-r", help="Output report file", required=False)
@click.option("--eval", "-ev", is_flag=True, help="Call pipeline evals", required=False)
def gen_graph_llm(
    env: str, cfg: str, dialogs: str, tgraph: str, output: str, report: str, eval: bool
):
    """Generate graph from dialogs data via d2g_llm pipeline"""
    load_dotenv(env)
    generate_llm(
        dialogs=dialogs,
        tgraph=tgraph,
        enable_evals=eval,
        config=cfg,
        graph_path=output,
        report_path=report,
    )


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogs", "-d", help="Input dialogs file", required=True)
@click.option("--graph", "-g", help="Input graph file", required=False)
@click.option("--tgraph", "-tg", help="Input true graph file", required=False)
@click.option("--output", "-o", help="Output graph file", required=False)
@click.option("--report", "-r", help="Output report file", required=False)
@click.option("--eval", "-ev", is_flag=True, help="Call pipeline evals", required=False)
def gen_graph_extender(
    env: str,
    cfg: str,
    dialogs: str,
    graph: str,
    tgraph: str,
    output: str,
    report: str,
    eval: bool,
):
    """Generate graph from dialogs data via d2g_llm pipeline"""
    load_dotenv(env)
    generate_extender(
        dialogs=dialogs,
        graph=graph,
        tgraph=tgraph,
        enable_evals=eval,
        config=cfg,
        graph_path=output,
        report_path=report,
    )
