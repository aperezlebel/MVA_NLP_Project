import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from ctcdecode import CTCBeamDecoder


class CustomDataset:
    def __init__(self, dataset_path, len_dataset, train_size, batch_size):
        self.dataset_path = dataset_path
        self.len_dataset = len_dataset
        self.train_size = train_size
        self.batch_size = batch_size
        self.it_train = 0
        self.it_eval = 0

    def get_next_train_batch(self):
        list_tens_audio, list_tens_labels = None, None
        if (self.it_train + 1) * self.batch_size <= self.train_size:
            list_tens_audio = [torch.load(self.dataset_path + '/audio/audio_' + str(i) + '.pt') for i in range(self.it_train * self.batch_size, (self.it_train + 1) * self.batch_size)]
            list_tens_labels = [torch.load(self.dataset_path + '/labels/label_' + str(i) + '.pt') for i in range(self.it_train * self.batch_size, (self.it_train + 1) * self.batch_size)]
            self.it_train += 1
        else:
            list_tens_audio = [torch.load(self.dataset_path + '/audio/audio_' + str(i) + '.pt') for i in range(self.it_train * self.batch_size, self.train_size)]
            list_tens_labels = [torch.load(self.dataset_path + '/labels/label_' + str(i) + '.pt') for i in range(self.it_train * self.batch_size, self.train_size)]
            self.it_train = 0
        input_lengths = torch.tensor([e.shape[1] for e in list_tens_audio])
        target_lengths = torch.tensor([e.shape[0] for e in list_tens_labels])
        targets = torch.cat(list_tens_labels)
        max_len = torch.max(input_lengths)
        for i in range(len(list_tens_audio)):
            length = list_tens_audio[i].shape[1]
            if length < max_len:
                list_tens_audio[i] = torch.cat((list_tens_audio[i], torch.zeros((1,max_len-length,768))), dim=1)
        X = torch.cat(list_tens_audio, dim=0)
        return X, input_lengths, targets, target_lengths

    def get_next_eval_batch(self):
        list_tens_audio, list_tens_labels = None, None
        if self.train_size + (self.it_eval + 1) * self.batch_size <= self.len_dataset:
            list_tens_audio = [torch.load(self.dataset_path + '/audio/audio_' + str(i) + '.pt') for i in range(self.train_size + self.it_eval * self.batch_size, self.train_size + (self.it_eval + 1) * self.batch_size)]
            list_tens_labels = [torch.load(self.dataset_path + '/labels/label_' + str(i) + '.pt') for i in range(self.train_size + self.it_eval * self.batch_size, self.train_size + (self.it_eval + 1) * self.batch_size)]
            self.it_eval += 1
        else:
            list_tens_audio = [torch.load(self.dataset_path + '/audio/audio_' + str(i) + '.pt') for i in range(self.train_size + self.it_eval * self.batch_size, self.len_dataset)]
            list_tens_labels = [torch.load(self.dataset_path + '/labels/label_' + str(i) + '.pt') for i in range(self.train_size + self.it_eval * self.batch_size, self.len_dataset)]
            self.it_eval = 0
        input_lengths = torch.tensor([e.shape[1] for e in list_tens_audio])
        target_lengths = torch.tensor([e.shape[0] for e in list_tens_labels])
        targets = torch.cat(list_tens_labels)
        max_len = torch.max(input_lengths)
        for i in range(len(list_tens_audio)):
            length = list_tens_audio[i].shape[1]
            if length < max_len:
                list_tens_audio[i] = torch.cat((list_tens_audio[i], torch.zeros((1,max_len-length,768))), dim=1)
        X = torch.cat(list_tens_audio, dim=0)
        return X, input_lengths, targets, target_lengths



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 61)

    def forward(self, x):
        x_drop = self.dropout(x)
        fc = self.fc(x_drop)
        output = F.log_softmax(fc, dim=2)
        return output



def train(model, device, dataset, n_epochs, learning_rate):
    ctc_loss = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for ep in range(n_epochs):
        print("Epoch:", ep)
        model.train()
        for it in range(dataset.train_size // dataset.batch_size + 1*(dataset.train_size % dataset.batch_size > 0)):
            X, input_lengths, targets, target_lengths = dataset.get_next_train_batch()
            X, input_lengths, targets, target_lengths = X.to(device), input_lengths.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            X = model(X).permute(1,0,2)
            loss = ctc_loss(X, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            print("It:", it, "Train loss:", loss.item())
        model.eval()
        mean_loss_eval = []
        with torch.no_grad():
            for it in range((dataset.len_dataset - dataset.train_size) // dataset.batch_size + 1*((dataset.len_dataset - dataset.train_size) % dataset.batch_size > 0)):
                X, input_lengths, targets, target_lengths = dataset.get_next_eval_batch()
                X, input_lengths, targets, target_lengths = X.to(device), input_lengths.to(device), targets.to(device), target_lengths.to(device)
                X = model(X).permute(1,0,2)
                loss = ctc_loss(X, targets, input_lengths, target_lengths)
                mean_loss_eval.append(loss.item())
        print("Average eval loss:", sum(mean_loss_eval)/len(mean_loss_eval))
        print("")

    os.makedirs('trained/', exist_ok=True)
    torch.save(model.state_dict(), f'trained/{model.__class__.__name__}-nepochs_{n_epochs}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLP')
    parser.add_argument('actions', nargs='+', type=str, help='Script to run.')
    # parser.add_argument('action', nargs='?', type=str, help='Action to run.')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.actions[0] == 'train':
        model = Net().to(device)
        dataset = CustomDataset('data_libri_en', 2607, 2200, 64)
        n_epochs = 15
        learning_rate = 0.001

        train(model, device, dataset, n_epochs, learning_rate)

    elif args.actions[0] == 'evaluate':
        model = Net().to(device)

        state_dict = torch.load(f'trained/{args.actions[1]}')
        model.load_state_dict(state_dict)

        dataset = CustomDataset('data_libri_en', 2607, 2200, 64)

        X, input_lengths, targets, target_lengths = dataset.get_next_train_batch()
        X, input_lengths, targets, target_lengths = X.to(device), input_lengths.to(device), targets.to(device), target_lengths.to(device)
        X = model(X)#.permute(1, 0, 2)

        print(X.shape)

        x = X[0, :, :]
        x = x[None, :, :]

        print(x.shape)

        output = x

        # decoder = CTCBeamDecoder(
        #     labels=[str(i) for i in range(61)],
        #     model_path=None,
        #     alpha=0,
        #     beta=0,
        #     cutoff_top_n=40,
        #     cutoff_prob=1.0,
        #     beam_width=100,
        #     num_processes=4,
        #     blank_id=0,
        #     log_probs_input=True,
        # )
        # beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)

        # print(beam_results.shape)
        # # print(beam_scores.shape)
        # best_beam = beam_results[:, 0, :out_lens[0][0]]
        # print(best_beam.shape)
        # beam_results[0][0][:out_len[0][0]]

        decoder = CTCBeamDecoder(
            [str(i) for i in range(61)],
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=4,
            blank_id=0,
            log_probs_input=True
        )

        model.eval()
        with torch.no_grad():
            X, input_lengths, targets, target_lengths = dataset.get_next_eval_batch()
            X, input_lengths, targets, target_lengths = X.to(device), input_lengths.to(device), targets.to(device), target_lengths.to(device)
            X = model(X)
            x = X[0, :input_lengths[0], :]
            print(x.shape)
            beam_results, beam_scores, timesteps, out_lens = decoder.decode(X[:1, :input_lengths[0], :])
            best_beam = beam_results[0][0][:out_lens[0][0]]
            print(best_beam)
            # print(best_beam.shape)
            # print(targets[:target_lengths[0]].shape)

            # print(x.shape)

            phones = torch.argmax(x, dim=1)
            # print(phones.shape)
            print(phones)

    elif args.actions[0] == 'examinate':
        model = Net().to(device)

        state_dict = torch.load(f'trained/{args.actions[1]}')
        model.load_state_dict(state_dict)

        dataset = CustomDataset('data_libri_en', 2607, 2200, 64)

        model.eval()
        with torch.no_grad():
            X, input_lengths, targets, target_lengths = dataset.get_next_eval_batch()
            X, input_lengths, targets, target_lengths = X.to(device), input_lengths.to(device), targets.to(device), target_lengths.to(device)
            X = model(X)
            x = X[0, :input_lengths[0], :]
            # print(x.shape)

            phones = torch.argmax(x, dim=1)
            # print(phones.shape)
            print(phones)
            print(phones[phones != 0])




    else:
        raise NotImplementedError(f'unkown script "{args.actions[0]}"')
