import view
import match
import sfm

import os
import pathlib

root_path = pathlib.Path(__file__).parent.parent
dataset_path = os.path.join(root_path, 'datasets')


def main()->None:
    views = view.create_views(dataset_path)
    matches = match.create_matches(views)
    print(views[0].name)

if __name__== "__main__":
    main()