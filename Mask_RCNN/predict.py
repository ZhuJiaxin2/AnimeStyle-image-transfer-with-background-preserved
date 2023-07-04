import os
import time
import json
import cv2 as cv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # ############################  测试单个模型的时候删去这条   ############################
    os.chdir('./Mask_RCNN')
    # ###################################################################################

    num_classes = 90  # 不包含背景
    box_thresh = 0.5
    weights_path = "./save_weights/mask_rcnn_weights.pth"
    #img_path = "./test.jpg"
    label_json_path = './coco91_indices.json'
    #anime_img_path = "./anime_test.jpg"

    current_dir = os.path.abspath('.')
    parent_dir = os.path.dirname(current_dir)
    public_dir = os.path.join(parent_dir, 'public')
    result_dir = os.path.join(parent_dir, 'result')

    input_dir = os.path.join(parent_dir, 'input')
    #input_dir = "./in/input_rcnn"
    output_dir = result_dir
    #output_dir = "./out/output_rcnn"
    human_dir = "./out/human_mask"
    anime_dir = public_dir
    #anime_dir = "./in/anime_rcnn"

    #num_human = 1

    # get devices
    device = torch.device('cpu')
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    ############################    load input dir   ################################
    os.makedirs(output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png",".jpeg", ".JPG", ".PNG", ".JPEG"]:
            continue
        img_path = os.path.join(input_dir, image_name)
        anime_img_path = os.path.join(anime_dir, image_name)
        human_path = os.path.join(human_dir, f'{image_name}_human.jpg')
        output_path = os.path.join(output_dir, f'{image_name}.jpg')

        # load image
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            #predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            # mask = Image.fromarray(predict_mask.astype(np.uint8)).convert('P')
            # mask.save('mask.png')

            # if len(predict_boxes) == 0:
            #     print("没有检测到任何目标!")
            #     return

            # plot_img = draw_objs(original_img,
            #                      boxes=predict_boxes,
            #                      classes=predict_classes,
            #                      scores=predict_scores,
            #                      masks=predict_mask,
            #                      category_index=category_index,
            #                      line_thickness=3,
            #                      font='arial.ttf',
            #                      font_size=20)
            # plt.imshow(plot_img)
            # plt.show()
            # # 保存预测的图片结果
            # plot_img.save("test_result.jpg")

            # ############################    输出各个实体的mask   ################################
            # # 假设您的模型预测结果存储在变量prediction中
            # # 您可以遍历所有掩膜并将其保存为图像
            # for i in range(len(predictions['masks'])):
            #     mask = predictions['masks'][i, 0]
            #     mask = mask.mul(255).byte().cpu().numpy()
            #     cv.imwrite(f'mask_{i}.png', mask)

            # ##########################    输出原图中score>90%的人物数量   ##############################
            # 假设你已经对图片进行了预测，得到了预测结果

            # 获取预测结果中的 labels 和 scores
            labels = predictions['labels']
            scores = predictions['scores']

            # 计算 score 大于 90% 的 person 的数量
            num_human = ((labels == 1) & (scores > 0.98)).sum().item()

            ############################    输出所有人物的mask   ################################
            # 假设您的模型预测结果存储在变量prediction中
            # 假设您想要将类别为1（human）的掩膜放在一张图像上
            class_id = 1

            # 获取类别为class_id的掩膜索引
            mask_indices = (predictions['labels'] == class_id).nonzero().view(-1)

            if num_human==1:
                # 获取第一个掩膜并将其转换为numpy数组
                mask = predictions['masks'][mask_indices[0], 0].mul(255).byte().cpu().numpy()
            # elif num_human==0:
            #     return
            else:
                # 获取第一个掩膜并将其转换为numpy数组
                mask = predictions['masks'][mask_indices[0], 0].mul(255).byte().cpu().numpy()
                # 遍历剩余的掩膜并将它们添加到第一个掩膜上
                # for i in range(1, len(mask_indices)):
                for i in range(1, num_human):
                    mask += predictions['masks'][mask_indices[i], 0].mul(255).byte().cpu().numpy()

            # 将累加后的掩膜保存为图像
            cv.imwrite(human_path, mask)
            #cv.imwrite(f'mask_{class_id}.png', mask)



            ############################    分割背景   ################################
            person = cv.imread(anime_img_path)
            back = cv.imread(img_path)
            # 这里将mask图转化为灰度图
            mask = cv.imread(human_path, cv.IMREAD_GRAYSCALE)
            # 将anime图resize到和原图一样的尺寸
            person = cv.resize(person, (back.shape[1], back.shape[0]))
            # 这一步是将背景图中的人像部分抠出来，也就是人像部分的像素值为0
            scenic_mask = ~mask
            scenic_mask = scenic_mask / 255.0
            back[:, :, 0] = back[:, :, 0] * scenic_mask
            back[:, :, 1] = back[:, :, 1] * scenic_mask
            back[:, :, 2] = back[:, :, 2] * scenic_mask
            # 这部分是将我们的人像抠出来，也就是背景部分的像素值为0
            mask = mask / 255.0
            person[:, :, 0] = person[:, :, 0] * mask
            person[:, :, 1] = person[:, :, 1] * mask
            person[:, :, 2] = person[:, :, 2] * mask
            # 这里做个相加就可以实现合并
            result = cv.add(back, person)
            cv.imwrite(output_path, result)


            # ############################    分割人像   ################################
            # # 假设您的原始图像存储在变量image中
            # # 假设您的掩膜图像存储在变量mask中
            # image = cv.imread(img_path)
            # mask = cv.imread('human.png', 0)
            #
            # # 将掩膜二值化
            # # _, mask = cv.threshold(mask, 0, 1, cv.THRESH_BINARY)
            # _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
            #
            # # 使用mask分割出人像部分
            # human = cv.bitwise_and(image, image, mask=mask)
            # # 保存人像图片
            # cv.imwrite('allhuman.png', human)




            # # 查找mask中的轮廓
            # contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            #
            # # 找到最大的轮廓
            # max_contour = max(contours, key=cv.contourArea)
            #
            # # 创建一个空白mask
            # new_mask = np.zeros_like(mask)
            #
            # # 绘制最大的轮廓
            # cv.drawContours(new_mask, [max_contour], -1, 255, -1)
            #
            # # # 腐蚀mask
            # # kernel = np.ones((5, 5), np.uint8)
            # # mask = cv.erode(mask, kernel, iterations=1000)
            #
            # # 使用新的mask分割出人像部分
            # human = cv.bitwise_and(image, image, mask=new_mask)
            #
            # # 保存人像图片
            # cv.imwrite('allhuman.png', human)




            # # 将掩膜扩展为与原始图像具有相同的形状
            # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            #
            # # 使用掩膜从原始图像中分割出人像
            # segmented_image = original_img * mask
            #
            # # 显示分割后的图像
            # cv.imshow('Segmented Image', segmented_image)
            # cv.waitKey(0)


if __name__ == '__main__':
    main()

