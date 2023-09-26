import json
import argparse

parser = argparse.ArgumentParser(description="get train/test coco annotations file")

args = parser.add_argument("--annotations", type=str, default="instances_test2017.json")

args = parser.parse_args()

project_dict = {"010": "Textbooks", "020": "Newspapers", "030": "Magazines", "040": "Test papers",
                "050": "Scientific articles", "060": "Notes"}


def add_train_domain(jsonfile, save_path):
    with open(jsonfile, "r") as jf:
        info = json.load(jf)
        for image in info["images"]:
            filename = image["file_name"]
            if filename[0] == "2":
                file_cate = "Books(photo)"
            else:
                file_cate = project_dict[filename.split('_')[0][:3]]
            image.update({"domain": file_cate})
            print(image)

    with open(save_path, "w") as jf_sv:
        json.dump(info, jf_sv)


if __name__ == "__main__":
    savepath = args.annotations[:-5] + "_document-type.json"
    add_train_domain(args.annotations, savepath)
