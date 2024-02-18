import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
import random,os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
import argparse


class Texts(Dataset):
    def __init__(self,data,max_length=1024) :
        super().__init__()
        tokenizer=GPT2TokenizerFast.from_pretrained('gpt2')
        self.texts=[]
        for row in data['text']:
            self.texts.append(torch.tensor(tokenizer.encode(f"{row[:max_length]}<|endoftext|>")))
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index) :
        return self.texts[index]


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, :]], dim=1)
        return packed_tensor, True, None


def train(
    dataset,
    model,
    batch_size=16,
    epochs=700,
    lr=2e-5,
    warmup_steps=200,
    save_model=False,
    save_dir=None
):
    device = torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None
    losses=[]
    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        print(loss)
        losses.append(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, _) = pack_tensor(entry, input_tensor, 768)
            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
    if save_model:
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"mymodel.pt"),
        )
        with open(os.path.join(save_dir, "loss"), "w") as file:
            file.write(f"epoch:{epoch}\nloss:")
            for item in losses:
                file.write(f"{item}\n")
    return model
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir")
    parser.add_argument("-s", "--modelsource")
    parser.add_argument("-t", "--tosave", action="store_true")
    parser.add_argument("-m", "--modeldir")
    args=parser.parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(args.modelsource)
    data=pd.read_csv(args.datadir)
    data=data[data['generated']==1].drop(['generated','Unnamed: 0'],axis=1)
    testdata=data.sample(frac=0.15,random_state=1)
    data=data[~data.index.isin(testdata.index)]
    data=Texts(data)
    train(data,model,save_model=args.tosave,save_dir=args.modeldir)