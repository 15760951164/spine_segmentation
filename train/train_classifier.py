import sys
sys.path.append("./")
from torch.utils.tensorboard import SummaryWriter
import random
from utils.cmtx import get_confusion_matrix, add_confusion_matrix, add_roc_curve
from sklearn.metrics import precision_recall_fscore_support
import json
import argparse
from models.ResNet import generate_resnet_model
from train.train_base import TrainLoop, default_data_extraction_func
from utils.sitk_np import sitk_to_npimage, npimage_to_sitk
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import glob
import torch
import numpy as np
import os

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class group_dataset(Dataset):

    def __init__(self, data_dir):
        super(group_dataset, self).__init__

        self.filelist = []
        self.labellist = []

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                self.filelist.append(file)
                self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index]
        if label <= 7:
            label = 0
        elif label <= 19 and label >= 8:
            label = 1
        elif label >= 20 and label != 28:
            label = 2
        else:
            label = 1

        self.msk_cube = out
        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)

class cervical_dataset(Dataset):

    def __init__(self, data_dir):
        super(cervical_dataset, self).__init__

        train_ids = os.listdir(data_dir)

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                if bone_label > 7:
                    continue
                self.filelist.append(file)
                self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index] - 1

        self.msk_cube = out

        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class thoracic_dataset(Dataset):

    def __init__(self, data_dir):
        super(thoracic_dataset, self).__init__

        train_ids = os.listdir(data_dir)

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                if bone_label == 28:
                    self.filelist.append(file)
                    self.labellist.append(bone_label)
                elif bone_label < 8 or bone_label > 19:
                    continue
                else:
                    self.filelist.append(file)
                    self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index]

        if label == 28:
            label = 11
        else:
            label -= 8

        self.msk_cube = out
        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class lumbar_dataset(Dataset):

    def __init__(self, data_dir):
        super(lumbar_dataset, self).__init__

        train_ids = os.listdir(data_dir)

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                if bone_label < 20 or bone_label == 28:
                    continue
                self.filelist.append(file)
                self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index]

        if label == 25:
            label = 4
        else:
            label -= 20

        self.msk_cube = out
        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class fracture_dataset(Dataset):

    def __init__(self, data_dir, train, split=0.75, norm=True):
        super(fracture_dataset, self).__init__

        self.filelist = []

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:

            if vol_id == "fracture":
                fracture_files = glob.glob(
                    os.path.join(data_dir, vol_id, "*bone*.npz"))
                if train:
                    fracture_files = fracture_files[0:int(
                        len(fracture_files)*split)]
                else:
                    fracture_files = fracture_files[int(
                        len(fracture_files)*split):]

            elif vol_id == "normal":
                normal_files = glob.glob(os.path.join(
                    data_dir, vol_id, "*bone*.npz"))

                if train:
                    normal_files = normal_files[0:int(len(normal_files)*split)]
                else:
                    normal_files = normal_files[int(len(normal_files)*split):]

        if len(fracture_files) * 1.2 > len(normal_files):
            n = int(len(fracture_files))
        else:
            n = int(len(fracture_files) * 1.3)

        normal_files = random.sample(normal_files, n)

        for file in fracture_files:
            self.filelist.append({"label": 0, "image_path": file})

        for file in normal_files:
            self.filelist.append({"label": 1, "image_path": file})

        t = []

        for file_dict in self.filelist:
            path = file_dict["image_path"]
            if "bone20" in path or "bone21" in path or "bone22" in path or "bone23" in path or "bone24" in path:
                t.append(file_dict)

        self.filelist = t

        pass

    def __getitem__(self, index):

        image_path = self.filelist[index]["image_path"]
        label = self.filelist[index]["label"]

        img_cube_array = np.load(image_path)["arr_0"]

        img_cube_array = torch.from_numpy(img_cube_array).to(
            dtype=torch.float32)
        self.img_cube = img_cube_array

        self.label = torch.tensor(label).to(torch.long)

        return self.img_cube, self.label, str(image_path)

    def __len__(self):
        return len(self.filelist)


class classifier_train(TrainLoop):

    def __init__(self,
                 model,
                 loss_function,
                 train_dataloader,
                 lr=0.001,
                 data_extraction_func=default_data_extraction_func,
                 test_dataloader=None,
                 optimizer=None,
                 model_save_path=None,
                 model_load_path=None,
                 model_pretrain_path=None,
                 max_iter=150,
                 checkpoint_iter=20,
                 weight_init_type="xavier",
                 device=1) -> None:
        super().__init__(model, loss_function, train_dataloader, lr,
                         data_extraction_func, test_dataloader, optimizer,
                         model_save_path, model_load_path, model_pretrain_path, max_iter,
                         checkpoint_iter, weight_init_type, device)

    def calculate_accuracy(self, outputs, targets):

        with torch.no_grad():
            batch_size = targets.size(0)

            _, pred = outputs.topk(1, 1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1))
            n_correct_elems = correct.float().sum().item()

            return pred, n_correct_elems / batch_size

    def calculate_precision_and_recall(self, outputs, targets):
        with torch.no_grad():
            _, pred = outputs.topk(1, 1, largest=True, sorted=True)
            precision, recall, fscoer, _ = precision_recall_fscore_support(
                targets.view(-1, 1).cpu().numpy(),
                pred.cpu().numpy(), zero_division=0, average="weighted")

            return precision, recall, fscoer

    @torch.no_grad()
    def test_step(self):
        self.model.eval()

        epoch_loss = 0
        accuracies = 0
        precisiones = 0
        recalles = 0
        fscoeres = 0

        preds = []
        labels = []

        score_list = []     # 存储预测得分
        label_list = []     # 存储真实标签

        if self.test_dataloader is not None:

            for i, batch in enumerate(self.test_dataloader):
                # model inputs
                input, label = self.data_extraction_func(batch)

                if self.device is None:
                    input = input.cuda()
                    label = label.cuda()
                else:
                    input = input.to(self.device)
                    label = label.to(self.device)

                predicted_label = self.model(input)
                loss = self.loss_function(predicted_label, label)
                epoch_loss += loss.item()

                pred, acc = self.calculate_accuracy(predicted_label, label)
                precision, recall, fscoer = self.calculate_precision_and_recall(
                    predicted_label, label)

                preds.append(pred.view(-1).cpu())
                labels.append(label.cpu())

                score_list.extend(predicted_label.cpu().numpy())
                label_list.extend(label.cpu().numpy())

                accuracies += acc
                precisiones += precision
                recalles += recall
                fscoeres += fscoer

                print('test epoch = {}, iter = {}/{}, loss = {}'.format(
                    self.current_iter, i, len(self.test_dataloader)-1, loss))

        total = len(self.test_dataloader)
        tb_writer.add_scalar('val/loss', epoch_loss, self.current_iter)
        tb_writer.add_scalar('val/acc', accuracies, self.current_iter)

        tb_writer.add_scalar('val/precision', precisiones /
                             total, self.current_iter)
        tb_writer.add_scalar('val/recall', recalles/total, self.current_iter)
        tb_writer.add_scalar('val/fscoer', fscoeres/total, self.current_iter)

        class_name = [str(i) for i in range(3)]

        cmtx = get_confusion_matrix(preds, labels, num_classes=len(class_name))
        add_confusion_matrix(tb_writer, cmtx, num_classes=len(class_name),
                             global_step=self.current_iter, tag="val/cmtx", class_names=class_name)
        add_roc_curve(score_list, label_list, tb_writer, "val/ROC",
                      self.current_iter, num_classes=len(class_name), class_names=class_name,
                      output_path=os.path.join(log_dir, f"{self.current_iter}_epoch.xlsx"))

        return epoch_loss

def get_class_weight(group_level, label_count_path):

    with open(label_count_path, 'r') as f:
        data = f.read()
        vert_count = json.loads(data)

    if group_level == 'group':
        arr = np.zeros(3)
        for l in range(1, 8):
            arr[0] += vert_count[str(l)]
        for l in range(8, 20):
            arr[1] += vert_count[str(l)]
        for l in range(20, 25):
            arr[2] += vert_count[str(l)]
        weight = 1.0/arr

    elif group_level == 'cervical':
        arr = np.zeros(7)
        for l in range(1, 8):
            arr[l-1] += vert_count[str(l)]
        weight = 1.0/arr

    elif group_level == 'thoracic':
        arr = np.zeros(12)
        for l in range(8, 20):
            arr[l-8] += vert_count[str(l)]
        weight = 1.0/arr

    elif group_level == 'lumbar':
        arr = np.zeros(5)
        for l in range(20, 25):
            arr[l-20] += vert_count[str(l)]
        weight = 1.0/arr

    return torch.from_numpy(weight).to(torch.float32).cuda(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_dir',
        type=str,
        default="/mnt/e/wyh/vertbrae/model_weight/classifier_SGD/",
    )

    parser.add_argument('--classify_level',
                        type=str,
                        default="group",
                        help='group | cervical | thoracic | lumbar | fracture')

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--train_dataset_dir',
                        type=str,
                        default="/mnt/e/wyh/vertbrae/train/data/classifier/train")

    parser.add_argument('--test_dataset_dir',
                        type=str,
                        default="/mnt/e/wyh/vertbrae/train/data/classifier/test")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--workers', type=int, default=12)


    args = parser.parse_args()
    label_count_path = "classifier_num_of_each_label.json"
    
    log_dir = os.path.join(args.save_dir, args.classify_level)
    check_dir(log_dir)
    tb_writer = SummaryWriter(log_dir)

    if args.classify_level == 'group':
        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=3)
        class_weights = get_class_weight('group', label_count_path)
        train_dataset = group_dataset(data_dir=args.train_dataset_dir)
        test_dataset = group_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == 'cervical':  # 颈椎
        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=7)
        class_weights = get_class_weight('cervical', label_count_path)
        train_dataset = cervical_dataset(data_dir=args.train_dataset_dir)
        test_dataset = cervical_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == 'thoracic':  # 腰椎

        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=12)
        class_weights = get_class_weight('thoracic', label_count_path)
        train_dataset = thoracic_dataset(data_dir=args.train_dataset_dir)
        test_dataset = thoracic_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == 'lumbar':  # 胸椎

        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=5)
        class_weights = get_class_weight('lumbar', label_count_path)
        train_dataset = lumbar_dataset(data_dir=args.train_dataset_dir)
        test_dataset = lumbar_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == "fracture":

        class_weights = torch.zeros(2)
        class_weights[0] = 1.2
        class_weights[1] = 1

        class_weights = class_weights.to(torch.float32).cuda(0)

        model = generate_resnet_model(n_input_channels=2,
                                      model_depth=50,
                                      n_classes=2)

        train_dataset = fracture_dataset(
            data_dir=args.train_dataset_dir, train=True, split=0.75)
        test_dataset = fracture_dataset(
            data_dir=args.test_dataset_dir, train=False, split=0.75)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=args.workers,
                                  drop_last=False,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=args.workers,
                                 drop_last=False,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                dampening=0,
                                weight_decay=1e-4,
                                nesterov=True)

    loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    main_train = classifier_train(model=model,
                                  loss_function=loss,
                                  train_dataloader=train_dataloader,
                                  lr=args.lr,
                                  test_dataloader=test_dataloader,
                                  optimizer=optimizer,
                                  model_load_path=None,
                                  model_save_path=os.path.join(args.save_dir, args.classify_level),
                                  max_iter=args.n_epoch)
    print(args)
    main_train.run()
