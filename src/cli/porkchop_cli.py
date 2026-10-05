from astrodynamics.porkchop_plot import porkchop
import argparse

def format_time(time_str: str) -> dt.datetime:
    split_str = time_str.split("/")
    day_str = split_str[0]
    month_str = split_str[1]
    year = int(split_str[2])
    
    # validation
    if len(day_str) != 2: 
        raise ValueError("Day must be in format 'XX'")
    if len(month_str) != 2: 
            raise ValueError("Day must be in format 'XX'")
    
    day = int(day_str)
    month = int(month_str)
    
    if day < 1 or day > 31:
        raise ValueError("Enter valid value for the day")
    if month < 1  or month > 12:
        raise ValueError("Enter valid value for the month")
    
    initial_dep_time = dt.datetime(year, month, day)
    
    return initial_dep_time



def porkchop_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-time", type=str, default="01/01/2017", help="The initial departure date in the format DD/MM/YYYY")
    parser.add_argument("--target", type=str, required=True, help="The target planet")
    parser.add_argument("--origin", type=str, default="earth", help="The origin planet")
    parser.add_argument("--N", type=int, default=500, help="The number of departure and arrival dates. Determines the number of points")
    parser.add_argument("--N-syn", type=int, default=1, help="The number of synodic periods to make the plot over")
    parser.add_argument("--save-fig", action="store_true", help="Whether or not to save an image of the plot in the outputs folder")
    parser.add_argument("--plot", action="store_true", help="Whether or not to immediately display the plot as a popup")

    a = parser.parse_args()
    
    initial_dep_time = format_time(a.init_time)
    
    porkchop(initial_dep_time, a.target, body1_name=a.origin, N=a.N, N_syn=a.N_syn, savefig=a.save_fig, plot=a.plot)
