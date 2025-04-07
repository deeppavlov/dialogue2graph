import click
<<<<<<< HEAD
from pathlib import PosixPath
=======
>>>>>>> dev
from dotenv import load_dotenv
from .commands.generate_data import generate_data
from .commands.generate_graph_light import generate_light
from .commands.generate_graph_llm import generate_llm
from .commands.generate_graph_extender import generate_extender


@click.group()
def cli():
    """Dialogue2Graph CLI tool for generating dialogue graphs and data"""
    pass


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--topic", "-t", help="Topic for dialogue generation", required=True)
@click.option("--output", "-o", help="Output file path", default="generated_data.json")
def gen_data(env: str, cfg: str, topic: str, output: str):
    """Generate dialogue data for a given topic"""
    load_dotenv(env)
    generate_data(topic, cfg, output)


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogues", "-d", help="Input dialogues file", required=True)
@click.option("--tgraph", "-t", help="Input true graph file", required=True)
@click.option("--output", "-o", help="Output graph file", required=True)
def gen_graph_light(env: str, cfg: str, dialogues: PosixPath, graph: PosixPath, tgraph: PosixPath, output: PosixPath):
    """Generate graph from dialogues data via d2g_algo pipeline"""
    load_dotenv(env)
<<<<<<< HEAD
    generate_light(dialogues, graph, tgraph, cfg, output)
=======
    generate_algo(dialogues, cfg, output)
>>>>>>> dev


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogues", "-d", help="Input dialogues file", required=True)
@click.option("--tgraph", "-t", help="Input true graph file", required=True)
@click.option("--output", "-o", help="Output graph file", required=False)
def gen_graph_llm(env: str, cfg: str, dialogues: PosixPath, tgraph: PosixPath, output: PosixPath):
    """Generate graph from dialogues data via d2g_llm pipeline"""
    load_dotenv(env)
<<<<<<< HEAD
    generate_llm(dialogues, tgraph, cfg, output)
=======
    generate_llm(dialogues, cfg, output)
>>>>>>> dev


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogues", "-d", help="Input dialogues file", required=True)
@click.option("--graph", "-g", help="Input graph file", required=True)
@click.option("--output", "-o", help="Output graph file", required=True)
def gen_graph_extender(env: str, cfg: str, dialogues: PosixPath, output: PosixPath):
    """Generate graph from dialogues data via d2g_llm pipeline"""
    load_dotenv(env)
    generate_extender(dialogues, cfg, output)


if __name__ == "__main__":
    cli()
