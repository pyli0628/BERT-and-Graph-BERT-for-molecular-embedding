import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm


class Trainer:

    def __init__(self, option, gat, model, train_dataloader, test_dataloader=None):

        # Superparam
        lr = option.lr
        betas = (option.adam_beta1, option.adam_beta2)
        weight_decay = option.adam_weight_decay
        with_cuda = option.with_cuda
        cuda_devices = option.cuda_devices
        log_freq = option.log_freq

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:{}".format(cuda_devices[0]) if cuda_condition else "cpu")

        self.gat = gat
        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices) > 1:
            print("Using %d GPUS for train" % len(cuda_devices))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader


        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        if train:
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}

                output = self.model.forward(data['x'],data['adj'])

                loss = self.criterion(output.transpose(1,2), data["label"])

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                avg_loss += loss.item()
                correct = self.cal_acc(output, data['label'])
                total_correct+=correct


                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                    'correct':correct
                }

                # if i % self.log_freq == 0:
                #     data_iter.write(str(post_fix))

        else:
            self.model.eval()
            with torch.no_grad():
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    output = self.model.forward(data['x'], data['adj'])

                    loss = self.criterion(output.transpose(1, 2), data["label"])

                    avg_loss += loss.item()
                    correct = self.cal_acc(output, data['label'])
                    total_correct += correct

                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss.item(),
                        'correct': correct
                    }

                    # if i % self.log_freq == 0:
                    #     data_iter.write(str(post_fix))


        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss /
                len(data_iter),'acc:{}'.format(total_correct/(len(data_iter))))

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "ep%d" % epoch
        torch.save(self.gat.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    def cal_acc(self,output,label):
        bs,l,v = output.size()
        mask = label.gt(0).repeat(1,v).view(bs,v,l).permute(0,2,1)
        count=0
        for i in range(bs):
            if mask[i].any():

                out_i =\
                torch.masked_select(output[i],mask[i]).view(-1,v).argmax(dim=-1).squeeze()
                label_i = torch.masked_select(label[i],label.gt(0)[i]).squeeze()
                if out_i.equal(label_i):
                    count+=1
        return count/bs

        
class Trainer2:

    def __init__(self, option, gat, model, train_dataloader, test_dataloader=None):

        # Superparam
        lr = option.lr
        betas = (option.adam_beta1, option.adam_beta2)
        weight_decay = option.adam_weight_decay
        with_cuda = option.with_cuda
        cuda_devices = option.cuda_devices
        log_freq = option.log_freq

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:{}".format(cuda_devices[0]) if cuda_condition else "cpu")

        self.gat = gat
        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices) > 1:
            print("Using %d GPUS for train" % len(cuda_devices))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader


        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion1 = nn.NLLLoss()
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_next_cor = 0
        total_element =0
        if train:
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}

                out1,out2,out3 = self.model.forward(data['x1'],data['x2'],data['adj1'],data['adj2'])

                mask_loss = self.criterion(out1.transpose(1,2), data["label1"])+\
                    self.criterion(out2.transpose(1,2),data['label2'])
                next_loss = self.criterion1(out3,data['next_label'])
                loss = mask_loss+next_loss

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                avg_loss += loss.item()
                correct = self.cal_acc(out1, data['label1'])
                total_correct+=correct

                next_correct = out3.argmax(dim=-1).eq(data["next_label"]).sum().item()
                total_next_cor+=next_correct
                total_element+=data['next_label'].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                    'correct':correct,
                    'next correct':total_next_cor/total_element*100
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

        else:
            self.model.eval()
            with torch.no_grad():
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    out1, out2, out3 = self.model.forward(data['x1'], data['x2'], data['adj1'], data['adj2'])

                    mask_loss = self.criterion(out1.transpose(1, 2), data["label1"]) + \
                                self.criterion(out2.transpose(1, 2), data['label2'])
                    next_loss = self.criterion1(out3, data['next_label'])
                    loss = mask_loss + next_loss

                    # 3. backward and optimization only in train
                    if train:
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    avg_loss += loss.item()
                    correct = self.cal_acc(out1, data['label1'])
                    total_correct += correct

                    next_correct = out3.argmax(dim=-1).eq(data["next_label"]).sum().item()
                    total_next_cor += next_correct
                    total_element += data['next_label'].nelement()

                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss.item(),
                        'correct': correct,
                        'next correct': total_next_cor / total_element * 100
                    }


                    if i % self.log_freq == 0:
                        data_iter.write(str(post_fix))


        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss /
                len(data_iter),'acc:{}'.format(total_correct/(len(data_iter))))
        print('next acc=',total_next_cor/total_element*100 )

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "ep%d" % epoch
        torch.save(self.gat.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    def cal_acc(self,output,label):
        bs,l,v = output.size()
        mask = label.gt(0).repeat(1,v).view(bs,v,l).permute(0,2,1)
        count=0
        for i in range(bs):
            if mask[i].any():

                out_i =\
                torch.masked_select(output[i],mask[i]).view(-1,v).argmax(dim=-1).squeeze()
                label_i = torch.masked_select(label[i],label.gt(0)[i]).squeeze()
                if out_i.equal(label_i):
                    count+=1
        return count/bs

