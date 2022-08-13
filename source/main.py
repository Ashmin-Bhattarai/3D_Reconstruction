import view
import match
import sfm

import os
import pathlib

dataset_folder_name = 'fountain'

root_path = pathlib.Path(__file__).parent.parent
dataset_path = os.path.join(root_path, 'datasets' ,dataset_folder_name)


def main()->None:
    views = view.create_views(dataset_path)
    matches = match.create_matches(views)
    sfm_obj = sfm.SFM(views, matches)
    sfm_obj.reconstruct()
    
    print("\n*****************Done matching*****************")

if __name__== "__main__":
    main()