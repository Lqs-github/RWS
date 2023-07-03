import os
import time
import datetime
import torch
from src import deeplabv3_resnet50, deeplabv3_resnet50_combine
from train_utils import train_one_epoch, train_one_epoch_add_W, evaluate, create_lr_scheduler
from dataset import VOCSegmentation
import transforms as T
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3_resnet50_cam
import torch.functional as F
import numpy as np
import requests
import torchvision
import cv2
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from utils import caculate_n_f_map, slice, miss_red, SegmentationModelOutputWrapper, SemanticSegmentationTarget, SegmentationPresetTrain, SegmentationPresetEval

Thresh=0.95 # T
epochs=1 # Number of single training rounds
delta=20 # The range of small RoI convergence
# classes = [
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ] VOC 2012 20 class

names='car'

def get_transform(train):
    base_size = 520 #9G
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

def create_model(aux, num_classes, pretrain=True):
    model = deeplabv3_resnet50_combine(aux=aux, num_classes=num_classes)

    if pretrain:
        weights_dict = torch.load("./deeplabv3_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def main(args):
    times=1
    all_large_H=[]
    all_small_H=[]
    all_image2all_slice = []
    samll_img_num = 0
    while times <= 20: # The maximum cycle is 20 times
        if times!=1 and times%2==1:
            cur_samll_img_num = len(all_small_img_names)
            if cur_samll_img_num - samll_img_num < delta:
                break # Stop when the number of small RoI converges

        if times!=1 and times%2==1:
            all_large_H=[]
            all_small_H=[] 
            all_image2all_slice = [] 

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        batch_size = 4 #4
        # segmentation nun_classes + background
        num_classes = args.num_classes + 1

        # Used to save information during training and validation
        results_file = "results_times_"+str(times)+".txt"

        if times%2==1:
            # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
            train_dataset = VOCSegmentation(args.data_path,
                                            year="2012",
                                            transforms=get_transform(train=True),
                                            txt_name="train.txt")
        else:
            # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
            train_dataset = VOCSegmentation(args.data_path,
                                            year="2012",
                                            transforms=get_transform(train=True),
                                            txt_name="small.txt")

        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
        val_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=False),
                                    txt_name="val.txt")

        # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        num_workers = 1
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                collate_fn=val_dataset.collate_fn)

        model = create_model(aux=args.aux, num_classes=num_classes)
        model.to(device)

        params_to_optimize = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
        ]

        if args.aux:
            params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": args.lr * 10})

        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

        scaler = torch.cuda.amp.GradScaler() if args.amp else None

        # Create a learning rate update strategy, here it is updated once per step (not per epoch)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])
        
        start_time = time.time()

        model.eval()

        #RoI convergence stop
    
        for epoch in range(args.start_epoch, args.epochs):
            if times==1 or times%2==0:
                f_map = [0, 0, 0, 0]
                n = 1
                mean_loss, lr, all_layer_features = train_one_epoch_add_W(model, optimizer, train_loader, device, epoch, f_map, n,
                                                lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
            else:
                mean_loss, lr, all_layer_features = train_one_epoch_add_W(model, optimizer, train_loader, device, epoch, f_map, n,
                                                lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

            if times % 2==1:
                large_f = all_layer_features
            else:
                small_f = all_layer_features

            with torch.no_grad():
                confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
            val_info = str(confmat)

            print(val_info)
            # write into txt
            with open(results_file, "a") as f:
                # Record the corresponding train_loss, lr and each metric of the validation set for each epoch
                train_info = f"[epoch: {epoch}]\n" \
                            f"train_loss: {mean_loss:.4f}\n" \
                            f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

            save_file = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))
        
#HL

        if times%2==1:
            with open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",'r') as fff:
                content = fff.read().splitlines()
        else:
            with open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/small.txt",'r') as fff:
                content = fff.read().splitlines()
        path_image="./data/VOCdevkit/VOC2012/JPEGImages/"
        path_label="./data/VOCdevkit/VOC2012/SegmentationClass/"
        #print(content)  
        if times==1:
            first_content=content

        line_index=0
        
        all_small_img_names = []
        for line in content:
            
            if line == '':
                continue
            src=path_image+line+".jpg"
            images=cv2.imread(src)
            image=np.array(images)

            #The trained model
            weights_path = "./save_weights/model_"+str(epochs-1)+".pth"

            rgb_img = np.float32(image) / 255
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            model = deeplabv3_resnet50_cam(pretrained=False, progress=False)

            # delete weights about aux_classifier
            weights_dict = torch.load(weights_path, map_location='cpu')['model']
            for k in list(weights_dict.keys()):
                if "aux" in k:
                    del weights_dict[k]

            # load weights
            model.load_state_dict(weights_dict)

            model = model.eval()

            if torch.cuda.is_available():
                model = model.cuda()
                input_tensor = input_tensor.cuda()
            
            output = model(input_tensor)
            #print(type(output), output.keys())
                
            model = SegmentationModelOutputWrapper(model)
            output = model(input_tensor)

            normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
            sem_classes = [
                '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]
            sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

            car_category = sem_class_to_idx[names]

            car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
            car_mask_float = np.float32(car_mask == car_category)

            both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
            Image.fromarray(both_images)            

            all_grayscale_cam=[]
            #FL
            all_target_layers=[[model.model.backbone.layer1], [model.model.backbone.layer2], [model.model.backbone.layer3], [model.model.backbone.layer4]]            
            for target_layers in all_target_layers: 
                #target_layers = [model.model.backbone.layer4]           
                targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
                with GradCAM(model=model,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets)[0, :]
                    import copy
                    grayscale_cam =copy.copy(grayscale_cam)
                    all_grayscale_cam.append(grayscale_cam)
                #Label
                lab=cv2.imread(path_label+line+".png")
                img_array=np.array(lab)
                img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
                shape = img_array.shape
                
                for k in range(0,shape[0]):
                    for j in range(0,shape[1]):
                        value = img_array[k, j]
                        if value==0:
                            grayscale_cam[k,j]=0

                #HL
                Image.fromarray(grayscale_cam)

            [x_H,y_H]=all_grayscale_cam[0].shape
            maps=np.zeros((x_H,y_H))
            for xi in range(x_H):
                for yi in range(y_H):
                    # Maximum feature map
                    maps[xi,yi]=max([all_grayscale_cam[0][xi,yi],all_grayscale_cam[1][xi,yi],all_grayscale_cam[2][xi,yi],all_grayscale_cam[3][xi,yi]])
            if times%2==1:
                large_H =[all_grayscale_cam[0], all_grayscale_cam[1], all_grayscale_cam[2], all_grayscale_cam[3], maps]
                all_large_H.append(large_H)
            else:
                small_H = [all_grayscale_cam[0], all_grayscale_cam[1], all_grayscale_cam[2], all_grayscale_cam[3], maps]
                all_small_H.append(small_H)

            CAM_VIS = False

            if CAM_VIS and times%2==1:
                # Missing weights chart
                w1=maps-all_grayscale_cam[0]
                w2=maps-all_grayscale_cam[1]
                w3=maps-all_grayscale_cam[2]
                w4=maps-all_grayscale_cam[3]

                labs=lab/255
                # Information weighting
                cam_image_F1 = show_cam_on_image(labs, all_grayscale_cam[0], use_rgb=True)
                cam_image_F2 = show_cam_on_image(labs, all_grayscale_cam[1], use_rgb=True)
                cam_image_F3 = show_cam_on_image(labs, all_grayscale_cam[2], use_rgb=True)
                cam_image_F4 = show_cam_on_image(labs, all_grayscale_cam[3], use_rgb=True)    

                cam_image_w1 = show_cam_on_image(labs, w1, use_rgb=True)
                cam_image_w2 = show_cam_on_image(labs, w2, use_rgb=True)
                cam_image_w3 = show_cam_on_image(labs, w3, use_rgb=True)
                cam_image_w4 = show_cam_on_image(labs, w4, use_rgb=True)

                # Missing weights
                cam_image_w1=miss_red(cam_image_w1,img_array)
                cam_image_w2=miss_red(cam_image_w2,img_array)
                cam_image_w3=miss_red(cam_image_w3,img_array)
                cam_image_w4=miss_red(cam_image_w4,img_array)
                        
                path1="./Hot/"+str(times)+"/"
                if not os.path.exists(path1):
                    os.makedirs(path1)
                cv2.imwrite(path1+str(times)+"_f1_"+line+".png",cam_image_w1)
                cv2.imwrite(path1+str(times)+"_f2_"+line+".png",cam_image_w2)
                cv2.imwrite(path1+str(times)+"_f3_"+line+".png",cam_image_w3)
                cv2.imwrite(path1+str(times)+"_f4_"+line+".png",cam_image_w4)
                
                # path2="./gray/"+str(times)+"/"
                # if not os.path.exists(path2):
                #     os.makedirs(path2)
                # cv2.imwrite(path2+str(times)+"_f1_"+line+".png",all_grayscale_cam[0]*100)
                # cv2.imwrite(path2+str(times)+"_f2_"+line+".png",all_grayscale_cam[1]*100)
                # cv2.imwrite(path2+str(times)+"_f3_"+line+".png",all_grayscale_cam[2]*100)
                # cv2.imwrite(path2+str(times)+"_f4_"+line+".png",all_grayscale_cam[3]*100)
        
            if times%2==1:#Large size slices
                all_slice_K, small_img_names =slice(grayscale_cam,img_array,images,Thresh,path_image,path_label, line)#small RoI
                all_image2all_slice.append(all_slice_K)
                all_small_img_names.extend(small_img_names)

            line_index=line_index+1
        if times%2==0: # small  caculate f and n
            all_k = []
            all_index = []
            for i, samll_set in enumerate(all_image2all_slice):
                if len(samll_set)==0:
                    continue
                for j, each_samll in enumerate(samll_set):
                    all_k.append(each_samll)
                    all_index.append(i)

            all_indexes = [[k, index] for k,index in zip(all_k, all_index)]
            if len(all_indexes)%4==1:
                all_indexes.append(all_indexes[-1])

            f_map, n  = caculate_n_f_map(all_large_H, all_small_H, large_f, small_f, all_indexes)

            # pass
        if times==1 and len(all_small_img_names)==0:
            print("Already trained better, unable to improve.")
            break
        if times%2!=0 and times!=1 and len(all_small_img_names)==0:
            print("FWM lift complete.")   
            break
        if times%2==1:
            if times==1:
                small_img_num = len(all_small_img_names)
            with open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/small.txt",'w') as ff:
                i = 0
                for small_img_name in all_small_img_names:
                    ff.write(small_img_name) #List
                    i = i + 1
                if i%4==1:
                    ff.write(small_img_name)
        times=times+1
            

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")

    parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=epochs, type=int, metavar="N",#30
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
