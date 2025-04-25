from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_notebook_link(source: Path, destination: Path) -> None:
    """
    Create a symlink between two files.
    Used to create links to tutorials under docs/source/tutorials/ root.

    :param source: Path to source file (in tutorials/ dir).
    :param destination: Path to link file (in docs/source/tutorials/ dir).
    """
    destination.unlink(missing_ok=True)
    destination.parent.mkdir(exist_ok=True, parents=True)
    destination.symlink_to(source.resolve(), False)


def generate_doc_heading(name, link):
    main_str = f":doc:`{name} <./{link.replace('.ipynb', '')}>`"
    break_line = "~" * len(main_str)
    return f"{main_str}\n{break_line}"


def create_index_file(
    name2link: dict,
    destination: Path,
) -> None:
    """
    Create a package index file.
    Contains nbgalleries of files inside the package (and subpackages).

    :param included: A pair of package path and alias with or without list of subpackages.
    :param files: List of all tutorial links.
    :param destination: Path to the index file.
    """
    doc_headings = [
        generate_doc_heading(name, link) for name, link in name2link.items()
    ]
    doc_headings = "\n\n".join(doc_headings)
    toctree_entries = "\n\t\t".join(list(name2link.values()))
    title = 'Examples\n========'
    contents = f"""{title}

{doc_headings}

.. toctree::
    :hidden:
    {toctree_entries}
    """

    destination.parent.mkdir(exist_ok=True, parents=True)
    destination.write_text(contents)


def symlink_files_to_dest_folder(
    include: list[tuple[str, str] | tuple[str, str, list[tuple[str, str]]]] = None,
    source: str = "examples",
    destination: str = "docs/source/examples",
):
    include = [] if include is None else include
    logger.info(f"include {include}, {destination}")
    destination = Path(destination)

    name2link = {}
    for folder, name in include:
        current_dir = Path(f"{source}/{folder}")
        logger.info(f"current_dir {current_dir}")
        for entity in [
            obj for obj in set(current_dir.glob("./*")) if not obj.name.startswith("__")
        ]:
            base_name = f"{folder}.{entity.name}"
            if entity.is_file() and entity.suffix == ".ipynb":
                base_path = Path(base_name)
                logger.info(f"create_notebook_link {entity.name}")
                create_notebook_link(entity, destination / base_path)
                name2link[name] = str(base_path)

    path = destination / "index.rst"
    logger.info(f"dest {path}")
    create_index_file(name2link, path)
