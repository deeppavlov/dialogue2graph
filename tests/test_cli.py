import os
import pathlib

filepath = pathlib.Path(__file__).resolve().parent

def test_gen_llm():
    exit_status = os.system(f'dialogue2graph gen-graph-llm -c {filepath.joinpath("cfg.yml")} -d {filepath.joinpath("test_graph_1.json")} -t {filepath.joinpath("test_graph_1.json")}')
    assert exit_status == 0

def test_gen_light():
    exit_status = os.system(f'dialogue2graph gen-graph-light -c {filepath.joinpath("cfg.yml")} -d {filepath.joinpath("test_graph_1.json")} -t {filepath.joinpath("test_graph_1.json")}')
    assert exit_status == 0

def test_gen_extender():
    exit_status = os.system(f'dialogue2graph gen-graph-extender -c {filepath.joinpath("cfg.yml")} -d {filepath.joinpath("test_graph_1.json")} -t {filepath.joinpath("test_graph_1.json")}')
    assert exit_status == 0