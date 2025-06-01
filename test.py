from argparse import ArgumentParser
from torchsummary import summary
import cv2
import numpy as np
import torch
from model import SimpleCNN
import torch
import torch.nn as nn

def get_args():
    parser = ArgumentParser(description="Cnn inference")

    parser.add_argument('--images_path', '-ip', type=str, default=None)
    parser.add_argument('--images_size','-i', type=int, default=112, help='Size of Image')


    parser.add_argument('--checkpoints', '-c', type=str, default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    classes = ["damaged", "non_damaged"]



    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SimpleCNN(num_classes=2).to(device)
    summary(model, (3, 112, 112))

    if args.checkpoints:
        checkpoint = torch.load(args.checkpoints)
        model.load_state_dict(checkpoint["model"])

    else:
        print("No found checkpoint")
        exit(0)
    model.eval()

    ori_image = cv2.imread(args.images_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.images_size, args.images_size))
    image = np.transpose(image, (2,0,1))/255.0 #đổi thứ tự kênh và chuyển 0-255 ve` 0-1
    image = image[None,:,:,:] #thêm chiều batch_size 1 x 3 x 224 x 224
    image = torch.from_numpy(image).to(device).float() #chuển về tensor
    sofmax = nn.Softmax()

    with torch.no_grad(): #không tính grad
        output = model(image)
        probs = sofmax(output)

    max_idx = torch.argmax(probs)
    predicted_class = classes[max_idx]
    print("The test image is about {} with confident score of {:.2f}".format(predicted_class, probs[0, max_idx]))
    cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx] * 100), ori_image)
    cv2.waitKey(0)



