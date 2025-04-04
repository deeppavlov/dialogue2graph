import os
import pathlib
filepath = pathlib.Path(__file__).resolve().parent

def test_gen_llm():
    exit_status = os.system(f'dialogue2graph gen-graph-llm -c {filepath.joinpath("cfg.yml")} -d true_graph_1.json -t true_graph_1.json')
    assert exit_status == 0
