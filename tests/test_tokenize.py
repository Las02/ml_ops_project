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
        # Tokenize the data
        tokenized_data = self.tokenize_data.danish_tokenized["input_ids"][0].tolist()
        # Remove zero padding
        tokenized_data = [x for x in tokenized_data if x != 0]
        # Check if the tokenized data is not empty and contains integers
        assert len(tokenized_data) > 0
        assert all(isinstance(x, int) for x in tokenized_data)

        def test_dataloader(self):
            dataset = OpusDataset("data/test_data/test_data.txt")
            train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            # Get the first batch of data
            batch = next(iter(train_dataloader))
            input_ids = batch[1][0].tolist()
            # Remove zero padding
            input_ids = [x for x in input_ids if x != 0]
            # Assert input is correct by checking if the tokenized data is not empty and contains integers
            assert len(input_ids) > 0
            assert all(isinstance(x, int) for x in input_ids)

    def test_dataloader_full(self):
        dataset = OpusDataset("data/processed/train.txt")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        # Get the first batch of data

        breakpoint()
        for truth, input in dataloader:
            pass
            # outputs = model(input_ids=input, labels=truth)
            # preds = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)
            #
            # loss = outputs.loss
            # loss.backward()
            #
            # optimizer.step()
            # optimizer.zero_grad()
            #
            # # Remove "<pad>" from preds
            # preds_decoded = dataset.decode(preds)
            # preds_decoded = [pred.replace("<pad>", "") for pred in preds_decoded]


# def test_dataloader_detokenize():
#     dataset = OpusDataset("data/test_data/test_data.txt")
#     train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#     dataset.decode(next(iter(train_dataloader))[1])
