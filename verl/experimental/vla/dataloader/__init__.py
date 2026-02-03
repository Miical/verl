import importlib

def build_dataloader_components(
    dataset_type: str,
    repo_id: str,
    root: str,
):
    try:
        module = importlib.import_module(
            f"verl.experimental.vla.dataloader.{dataset_type}"
        )
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown dataset_type='{dataset_type}'. "
            f"Expected one of subfolders under dataloader/."
        ) from e

    make_dataset = getattr(module, "make_dataset", None)
    make_sampler = getattr(module, "make_sampler", None)
    make_collator = getattr(module, "make_collator", None)

    if make_dataset is None or make_collator is None:
        raise RuntimeError(
            f"dataloader.{dataset_type} must define make_dataset() and make_collator()"
        )
    
    dataset = make_dataset(repo_id=repo_id, root=root)
    sampler = make_sampler(dataset) if make_sampler is not None else None
    collator = make_collator()

    return dataset, sampler, collator
