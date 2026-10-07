from astro_tools.astrodynamics.porkchop_plot import porkchop
from astro_tools.cli.cli_tools import format_time
import argparse
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-time", type=str, default="01/01/2017", help="The initial departure date in the format DD/MM/YYYY")
    parser.add_argument("--target", type=str, required=True, help="The target planet")
    parser.add_argument("--origin", type=str, default="earth", help="The origin planet")
    parser.add_argument("--N", type=int, default=500, help="The number of departure and arrival dates. Determines the number of points")
    parser.add_argument("--N-syn", type=int, default=1, help="The number of synodic periods to make the plot over")
    parser.add_argument("--save-fig", type=Path, default=Path.cwd(), help="Whether or not to save an image of the plot, and if so where")
    parser.add_argument("--plot", action="store_true", help="Whether or not to immediately display the plot as a popup")

    args = parser.parse_args()
    
    initial_dep_time = format_time(args.init_time)
    
    b1_name = args.origin.lower()
    b2_name = args.target.lower()
    
    porkchop(initial_dep_time, b2_name, body1_name=b1_name, N=args.N, N_syn=args.N_syn, savepath=args.save_fig, plot=args.plot)

if __name__ == "__main__":
    main()