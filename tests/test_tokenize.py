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
        assert self.tokenize_data.danish_tokenized == "På Det Blandede EØS-Udvalgs vegne"
