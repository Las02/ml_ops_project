from ml_ops_project.data import Tokenize_data


class TestTokenizer:
    tokenize_data = Tokenize_data("data/test_data/test_data.txt")

    def test_read_data_size(self):
        assert len(self.tokenize_data.danish) == 8
        assert len(self.tokenize_data.english) == 8

    def test_read_data_input(self):
        assert self.tokenize_data.danish[0] == "På Det Blandede EØS-Udvalgs vegne"
        assert self.tokenize_data.english[0] == "For the EEA Joint Committee"

    def test_tokenize_data(self):
        # Assert tokenizer but ignore zero padding
        assert [x for x in self.tokenize_data.danish_tokenized["input_ids"][0].tolist() if x != 0] == [
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
