from datasets import load_dataset, Dataset, DatasetDict

def streamed_dataset_to_split(streamed, n_rows, test_size=0.2, seed=42):
    data = Dataset.from_list(list(streamed.take(n_rows)))  # take a small subset for testing
    splits = data.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(splits)

def load_finepdfs(n_rows):
    # Login using e.g. `huggingface-cli login` to access this dataset
    streamed = load_dataset("HuggingFaceFW/finepdfs", "eng_Latn", streaming=True, split="train")
    ds = streamed_dataset_to_split(streamed, n_rows=n_rows, test_size=0.2)  # use a small subset for testing

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
    ds["val"] = ds.pop("test")  # type: ignore

    print(ds)  # should show Dataset with 1000 rows

    return ds