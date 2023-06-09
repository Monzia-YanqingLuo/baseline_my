import os
import json
import shutil
from collections import defaultdict

class DataCreator:
    def __init__(self):
        self.data_directory = "/Users/yanqingluo/Desktop/HTCVS2/DATASETS"
        self.damage_intensity_encoding = defaultdict(lambda: 0)
        self.damage_intensity_encoding['destroyed'] = 3
        self.damage_intensity_encoding['major-damage'] = 2
        self.damage_intensity_encoding['minor-damage'] = 1
        self.damage_intensity_encoding['no-damage'] = 0

    def validate(self):
        if not os.path.exists(self.data_directory):
            raise ValueError(f"Folder does not exist at {self.data_directory}")

    def run(self):
        self.group_and_reorganize_images_by_damage("train")
        self.group_and_reorganize_images_by_damage("test")

    def group_and_reorganize_images_by_damage(self, subset):
        file_manager = shutil
        subset_directory = os.path.join(self.data_directory, subset)
        label_directory = os.path.join(subset_directory, "labels")
        grouped_data_directory = os.path.join(self.data_directory, "..", "split_data")
        image_directory = os.path.join(subset_directory, "images")

        for label_file in os.listdir(label_directory):
            if "_post" not in label_file:
                continue

            # Rest of the code

            damage = self.calculate_damage_level(os.path.join(label_directory, label_file))
            damage_encoded = self.damage_intensity_encoding[damage]
            image_copy_directory = os.path.join(grouped_data_directory, subset, str(damage_encoded))
            os.makedirs(image_copy_directory, exist_ok=True)

            image_file = os.path.splitext(label_file)[0] + ".png"

            file_manager.copy(
                os.path.join(image_directory, image_file),
                os.path.join(image_copy_directory, image_file)
            )

    def calculate_damage_level(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            features = data["features"]["lng_lat"]

            if not features:
                return "no-damage"

            damage_levels = []
            for feature in features:
                properties = feature["properties"]
                subtype = properties["subtype"]
                if subtype == "un-classified":
                    continue
                damage_levels.append(subtype)

            if not damage_levels:
                return "no-damage"

            return max(set(damage_levels), key=damage_levels.count)

def main():
    data_creator = DataCreator()
    data_creator.validate()
    data_creator.run()

if __name__ == "__main__":
    main()
