import pytest 
from src.ml_ops_project.data import split_data

# from torch.utils.data import Dataset
# from ml_ops_project.data import MyDataset

# def test_my_dataset():
#     """Test the MyDataset class."""
#     dataset = MyDataset("data/raw")
#     assert isinstance(dataset, Dataset)


def create_tmp_data(file_path, num_lines):
    """Create tmp data for testing."""
    with open(file_path, "w") as f:
        for i in range(num_lines):
            f.write(f"line {i}\n")

def test_split_data(tmp_path):
    """ Test split into training/test data according to percentage argument."""
    
    # create tmp data
    raw_path = tmp_path / "raw"
    processed_path = tmp_path / "processed"
    raw_path.mkdir(parents=True)
    processed_path.mkdir(parents=True)

    create_tmp_data(raw_path / "train.txt", 100)

    # split data
    split_data(raw_path, processed_path, 0.5)

    # check that the data has been split correctly according to percentage given
    with open(processed_path / "train.txt", "r") as f:
        train_lines = f.readlines()

    with open(processed_path / "test.txt", "r") as f:
        test_lines = f.readlines()

    assert len(train_lines) == 50
    assert len(test_lines) == 50

    





    







    

