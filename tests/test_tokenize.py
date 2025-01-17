from torch.utils.data import DataLoader

from ml_ops_project.data import OpusDataset, Tokenize_data


class TestTokenizer:
    tokenize_data = Tokenize_data("data/test_data/test_data.txt")

    def test_read_data_size(self):
        assert len(self.tokenize_data.danish) == 8
        assert len(self.tokenize_data.english) == 8

    def test_read_data_input(self):
        assert self.tokenize_data.danish[5] == "- Men hun siger nej."
        assert self.tokenize_data.english[5] == "But she's turned him down."

    def test_tokenize_data(self):
        # Assert tokenizer but ignore zero padding
        assert [
            x for x in self.tokenize_data.danish_tokenized["input_ids"][0].tolist() if x != 0
        ] == [
            276,
            2,
            374,
            17,
            272,
            16349,
            15,
            262,
            2,
            134,
            18,
            1265,
            26,
            2165,
            122,
            7,
            3,
            162,
            15141,
            1,
        ]


def test_dataloader():
    dataset = OpusDataset("data/test_data/test_data.txt")
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Assert input is correct
    assert [x for x in next(iter(train_dataloader))[1][0].tolist() if x != 0] == [
        276,
        2,
        374,
        17,
        272,
        16349,
        15,
        262,
        2,
        134,
        18,
        1265,
        26,
        2165,
        122,
        7,
        3,
        162,
        15141,
        1,
    ]


# def test_dataloader_detokenize():
#     dataset = OpusDataset("data/test_data/test_data.txt")
#     train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#     dataset.decode(next(iter(train_dataloader))[1])
