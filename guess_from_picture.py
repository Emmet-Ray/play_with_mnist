from PIL import Image
import numpy as np
import sys
import network

def load_image_into_numpy_array(path):
    # 打开图片并转换为灰度格式
    img = Image.open(path).convert('L')
    # 调整图片大小为28x28像素
    img = img.resize((28, 28))
    # 将图片数据转换为numpy数组
    img_array = np.array(img)
    #img_array = 255 - img_array
    # 将二维数组（28x28）转换为一维数组（784）
    img_array = img_array.flatten()
    # 将数组的数据类型转换为float32
    img_array = img_array.astype(np.float32)
    # 将数组的值范围从0-255转换为0-1
    img_array /= 255.0
    # 保存调整后的图像
    img = Image.fromarray((img_array * 255).reshape(28, 28).astype(np.uint8))
    img.save("adjust.jpg")
    return img_array

def guess(path, actual, use_trained):
    array = load_image_into_numpy_array(path)
    array = np.reshape(array, (784, 1))

    experiment_times = 10
    if use_trained == "true":
        trained = True
    else :
        trained = False

    print()
    if trained:
        print(f"***I have been trained. So I think I could make a good guess.***")
        experiment_times = 1; 
    else:
        print(f"***I have not been trained. So I would make random guesses.***")
    print()

    if not trained:
        count = 0
        while True:
            net = network.Network([784, 30, 10], load_from_file=trained)
            recognized_array = net.feedforward(array)
            recognized_num = np.argmax(recognized_array)
            count += 1
            if not (recognized_num == actual):
                print(f"I think it's {recognized_num}, wrong...") 
            else:
                if count > 10:
                    print(f"Finally! After {count} guesses, I got it. It's {actual}")
                else:
                    print(f"After {count} guesses, I got it. It's {actual}")
                break
    else:
        net = network.Network([784, 30, 10], load_from_file=trained)
        recognized_array = net.feedforward(array)
        recognized_num = np.argmax(recognized_array)
        print(f"I think it's {recognized_num}")
        if recognized_num == actual:
            print(f"I got it.")
        else:
            print(f"I made a bad decision.")
    

if not len(sys.argv) == 4:
    print(f"Usage: python guess_from_picture.py [picture_path] [actual_num] [use_trained]")
    print(f"[actual_num is num 0 to 9")
    print(f"[use_trained] above is either 'true' or 'false'.")
    print(f"Try again.")
    exit(0)

path = sys.argv[1]
actual_num = int(sys.argv[2])
use_trained = sys.argv[3]
guess(path, actual_num, use_trained)