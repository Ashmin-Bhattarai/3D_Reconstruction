import os
from pprint import pprint
from pathlib import Path

import match
import sfm
import view
from utils import sort_views

dataset_folder_name = "fountain"

root_path = Path(__file__).parent.parent
dataset_path = root_path / "datasets" / dataset_folder_name


def main() -> None:
    views = view.create_views(dataset_path)
    matches = match.create_matches(views)
    sorted_views = sort_views(matches)
    print(sorted_views)
    print([image.name for image in sorted_views])

    # pprint(next(iter(matches.items()))[1].number_of_inliers())
    # pprint(next(iter(matches.items()))[1].view1.__dict__)
    # pprint(next(iter(matches.items()))[1].view2.__dict__)
    # pprint(matches.items())
    # pprint(iter(matches).__dir__)
    # sfm_obj = sfm.SFM(views, matches)
    # sfm_obj.reconstruct()

    print("\n*****************Done matching*****************")


if __name__ == "__main__":
    main()
