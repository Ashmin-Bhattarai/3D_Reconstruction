import view
import match
from pympler import asizeof
# import sfm

import os
import sys
import pathlib

dataset_folder_name = 'japan'

root_path = pathlib.Path(__file__).parent.parent
dataset_path = os.path.join(root_path, 'datasets' ,dataset_folder_name)


def main()->None:
    views = view.create_views(dataset_path)
    print(f"Size of views: {asizeof.asizeof(views)}")
    matches = match.create_matches(views)
    # sfm_obj = sfm.SFM(views, matches)
    # sfm_obj.reconstruct()
    
    print("\n*****************Done matching*****************")

if __name__== "__main__":
    main()