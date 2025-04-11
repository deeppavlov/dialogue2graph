import os
import pathlib

filepath = pathlib.Path(__file__).resolve().parent

def test_gen_light():
    exit_status = os.system(f'dialogue2graph test-dg-gen -c {filepath.joinpath("cfg.yml")} -d {filepath.joinpath("test_cli.json")} -t {filepath.joinpath("test_cli.json")}')
    assert exit_status == 0
