import os
from ultralytics import YOLO
import click


@click.command()
@click.option("--source-dir", '-s', "source_dir", default="./", show_default=True)
def main(**kwargs):
    source_dir = kwargs["source_dir"]
    model_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    for model_file in model_files:
        model = YOLO(os.path.join(source_dir, model_file))
        print(f"Classes for {model_file}\n")
        print(model.names)
        print("\n\n")

if __name__ == "__main__":
    main()
