import astro_tools.engine.simulation as simulation
from astro_tools.cli.cli_tools import format_time
import subprocess, argparse
from importlib.resources import files

SIM_CONFIG_FPATH = files("astro_tools.engine").joinpath('config.yaml')


def select_editor(args):
    editor = "vim"

    if args.notepad: editor = "notepad"
    elif args.vim: editor = "vim"
    elif args.gvim: editor = "gvim"
    elif args.nano: editor = "nano"
    
    if sum([args.notepad, args.vim, args.gvim, args.nano]) > 1:
        print("Select only one editor")
        
    return editor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notepad", action="store_true", help="Edit config using notepad")
    parser.add_argument("--vim", action="store_true", help="Edit config using vim")
    parser.add_argument("--gvim", action="store_true", help="Edit config using gvim")
    parser.add_argument("--nano", action="store_true", help="Edit config using nano")
    
    args = parser.parse_args()
    
    editor = select_editor(args)
    
    subprocess.run([editor, str(SIM_CONFIG_FPATH)], check=True)
    
    sim = simulation.sim()
    sim.run() 
    
    
if __name__ == "__main__":
    main()