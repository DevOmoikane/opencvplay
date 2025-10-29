import click
import difPy

@click.command()
@click.option("--source-dir", "source_dir", default="./", show_default=True, help="Folder to search for images.")
def main(source_dir):
    dif = difPy.build(source_dir)
    duplicates = difPy.search(dif, similarity="duplicates")
    similar = difPy.search(dif, similarity="similar")
    try:
        duplicates.delete(silent_del=True)
    except Exception:
        pass
    try:
        similar.delete(silent_del=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
