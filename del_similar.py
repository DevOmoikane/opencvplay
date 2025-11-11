import click
import difPy
from rich import print

@click.command()
@click.option("--source-dir", '-s', "source_dir", default="./", show_default=True, help="Folder to search for images.")
@click.option("--compare-to", '-c', "destination_dir", default=None)
@click.option("--dry-run/--run", " /-r", default=True, show_default=True, help="Dry run.")
def main(source_dir, destination_dir, dry_run):
    dirs = [source_dir]
    if destination_dir:
        dirs.append(destination_dir)
    dif = difPy.build(dirs)
    similar = difPy.search(dif, similarity="similar", same_dim=False, chunksize=1000)
    try:
        if not dry_run:
            print("Deleting similar:")
            similar.delete(silent_del=True)
        else:
            print("Going to delete similar:")
            print(similar.result)
    except Exception:
        pass

if __name__ == "__main__":
    main()
