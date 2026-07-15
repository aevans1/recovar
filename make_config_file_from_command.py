import os

# So there's no gpu errors while only importing some argparse things
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import yaml
import argparse
import importlib

def make_config_file_from_command(command_name):
    
    # import the main function from the RECOVAR command
    module = importlib.import_module(f"recovar.commands.{command_name}")
    parser = module.build_parser() 
    config = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        config[action.dest] = {
            "default": action.default,
            "help": action.help,
            "type": getattr(action.type, "__name__", None),
        }
    with open(f"{command_name}.schema.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

def main():

    parser = argparse.ArgumentParser(description="give name of recovar command to make config file for")
    parser.add_argument("command_name", help="command name for recovar")
    args = parser.parse_args()
    make_config_file_from_command(args.command_name)

if __name__ == "__main__":
    main()