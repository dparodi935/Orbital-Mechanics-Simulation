import astro_tools.engine.simulation as simulation
import argparse
import datetime as dt
from pathlib import Path

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


def main() -> None:
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--save-fig", type=Path, default=Path.cwd(), help="Whether or not to save an image of the plot, and if so where")
    #parser.add_argument("--plot", action="store_true", help="Whether or not to immediately display the plot as a popup")

    #args = parser.parse_args()
    
    #initial_dep_time = format_time(args.init_time)
    
    sim = simulation.sim()
    sim.run() 
    
if __name__ == "__main__":
    main()