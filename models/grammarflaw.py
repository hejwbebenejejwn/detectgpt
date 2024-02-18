from transformers import GPT2TokenizerFast, GPT2Model
import torch, os


class GrammarFlaw(torch.nn.Module):
    def __init__(self, max_length=30):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding = GPT2Model.from_pretrained("gpt2").wte
        self.conv1 = torch.nn.Conv1d(
            in_channels=768, out_channels=20, kernel_size=4, padding=2
        )
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(kernel_size=4)
        self.fc = torch.nn.Linear(20, 2)
        for param in self.embedding.parameters():
            param.require_grad = False

    def forward(self, data, name="text"):
        texts = list(data.data[name])
        x = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        x = self.embedding(x["input_ids"])
        x.swapaxes_(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.max(dim=2)[0]
        x = self.fc(x)
        return x

    def trainfunc(self, data, optimizer, criterion, epochs):
        self.train()
        target = torch.tensor(list(data.data["right"]), dtype=torch.long)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(
                    "Train Epoch: {} \tLoss: {:.6f}".format(
                        epoch,
                        loss.item(),
                    )
                )

    def savemodel(self, dir="./cache/"):
        if not os.path.exists(dir):
            os.makedirs(dir)

        model_path = os.path.join(dir, "grammarflaw_model.pth")

        torch.save(self.state_dict(), model_path)

    def loadmodel(self, model_path="./cache/grammarflaw_model.pth"):
        self.load_state_dict(torch.load(model_path))
