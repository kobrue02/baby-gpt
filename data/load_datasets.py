from datasets import load_dataset, Dataset, DatasetDict

def streamed_dataset_to_split(streamed, n_rows, test_size=0.2, seed=42):
    """
    Convert a streamed dataset to a DatasetDict with train and test splits.
    """
    data = Dataset.from_list(list(streamed.take(n_rows)))  # take a small subset for testing
    splits = data.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(splits)

def load_finepdfs(n_rows):
    """
    map this to a Dataset of this shape:

    DatasetDict({
    train: Dataset({
            features: ['text'],
            num_rows: 8009762
    })
    val: Dataset({
            features: ['text'],
            num_rows: 4007
    })
    })
    """
    # Login using e.g. `huggingface-cli login` to access this dataset
    #streamed = load_dataset("HuggingFaceFW/finepdfs", "eng_Latn", streaming=True, split="train")
    #ds = streamed_dataset_to_split(streamed, n_rows=n_rows, test_size=0.2)  # use a small subset for testing
    ds = load_dataset("roneneldan/TinyStories")
    # map this to a Dataset of this shape:
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    ds = ds.remove_columns([col for col in ds['train'].column_names if col != 'text']) # type: ignore
    try: 
        ds["val"] = ds.pop("test")  # type: ignore
    except KeyError:
        ds["val"] = ds.pop("validation")
    print(ds)  # should show Dataset with 1000 rows
    return ds

def load_general_knowledge():
    ds = load_dataset("MuskumPillerum/General-Knowledge")
    # train/val split
    ds = ds["train"].train_test_split(test_size=0.2, seed=42) # type: ignore
    # remove empty rows
    ds = ds.filter(lambda x: x['Question'] != None and x['Answer'] != None)
    ds["val"] = ds.pop("test")  
    print(ds)
    return ds
