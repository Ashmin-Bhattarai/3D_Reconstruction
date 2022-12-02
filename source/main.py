import os
from pathlib import Path

import match
import sfm
import view
from utils import sort_images

dataset_folder_name = "fountain"

root_path = Path(__file__).parent.parent
dataset_path = root_path / "datasets" / dataset_folder_name


def main() -> None:
    views = view.create_views(dataset_path)
    matches = match.create_matches(views)
    sorted_images = sort_images(matches)
    print(sorted_images)

    # TODO
    # sort images ✔️
    # pprint(next(iter(matches.items()))[1].number_of_inliers())
    # sfm_obj = sfm.SFM(views, matches)
    # sfm_obj.reconstruct()

    print("\n*****************Done matching*****************")


if __name__ == "__main__":
    main()
