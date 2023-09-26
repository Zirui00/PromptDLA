import json
import argparse

parser = argparse.ArgumentParser(description="get train/test coco annotations file")

args = parser.add_argument("--annotations", type=str, default="test.json")

args = parser.parse_args()


def add_domain(jsonfile, savepath):
    with open(jsonfile, "r") as jf:
        info = json.load(jf)
    images = info["images"]
    for image in images:
        domain = image["doc_category"]
        image.update({"domain": domain})
        print(image)
    with open(savepath, "w") as jf_save:
        json.dump(info, jf_save)


def main():
    savepath = args.annotations[:-5] + "_document-type.json"
    add_domain(args.annotations, savepath)


if __name__ == "__main__":
    main()
