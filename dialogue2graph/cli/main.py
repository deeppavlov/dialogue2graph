import click
import yaml
from dotenv import load_dotenv
from .commands.generate_data import generate_data

# from .commands.generate_graph import generate_graph


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


# @cli.command()
# @click.option('--env', '-e', help='Path to .env file', default='.env')
# @click.option('--dialogues', '-d', help='Input dialogues file', required=True)
# @click.option('--output', '-o', help='Output graph file', required=True)
# def gen_graph(env: str, dialogues: str, output: str):
#     """Generate graph from dialogue data"""
#     load_dotenv(env)
#     generate_graph(dialogues, output)

if __name__ == "__main__":
    cli()
