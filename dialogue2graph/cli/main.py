import click
import yaml
from dotenv import load_dotenv
from .commands.generate_data import generate_data
from .commands.generate_graph_algo import generate_algo
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
    with open(cfg) as stream:
        config: dict = {}
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    generate_data(topic, config, output)


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogues", "-d", help="Input dialogues file", required=True)
@click.option("--output", "-o", help="Output graph file", required=True)
def gen_graph_algo(env: str, cfg: str, dialogues: str, output: str):
    """Generate graph from dialogues data via d2g_algo pipeline"""
    load_dotenv(env)
    with open(cfg) as stream:
        config: dict = {}
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    generate_algo(dialogues, config, output)


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogues", "-d", help="Input dialogues file", required=True)
@click.option("--output", "-o", help="Output graph file", required=True)
def gen_graph_llm(env: str, cfg: str, dialogues: str, output: str):
    """Generate graph from dialogues data via d2g_llm pipeline"""
    load_dotenv(env)
    with open(cfg) as stream:
        config: dict = {}
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    generate_llm(dialogues, config, output)


@cli.command()
@click.option("--env", "-e", help="Path to .env file", default=".env")
@click.option("--cfg", "-c", help="Path to cfg.yml file", default="cfg.yml")
@click.option("--dialogues", "-d", help="Input dialogues file", required=True)
@click.option("--output", "-o", help="Output graph file", required=True)
def gen_graph_extender(env: str, cfg: str, dialogues: str, output: str):
    """Generate graph from dialogues data via d2g_llm pipeline"""
    load_dotenv(env)
    with open(cfg) as stream:
        config: dict = {}
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    generate_extender(dialogues, config, output)


if __name__ == "__main__":
    cli()
