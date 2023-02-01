##

from sklearn.preprocessing import LabelBinarizer
from scipy.io import loadmat
from keras.datasets import mnist, cifar10 , fashion_mnist, cifar100
from keras.models import load_model
from keras.utils import np_utils
##
import os
import argparse
import keras
from keras.callbacks import ModelCheckpoint
from termcolor import colored
from ATS.ATS import ATS
from utils import model_conf
import numpy as np

from utils.utils import num_to_str, shuffle_data, shuffle_data3


def color_print(s, c):
    print(colored(s, c))


def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=10, verbose=1):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu


def diverse_errors_num(y_s, y_psedu):
    fault_pair_arr = []
    fault_idx_arr = []
    for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            key = (y_s_temp, y_psedu_temp)
            if key not in fault_pair_arr:
                fault_pair_arr.append(key)
                fault_idx_arr.append(ix)
    return len(fault_idx_arr)


def get_tests(x_dau, y_dau, order):
    x_sel, y_sel = x_dau[:test_size // 2], y_dau[:test_size // 2]
    order1=order[:test_size // 2]
    x_val, y_val = x_dau[test_size // 2:], y_dau[test_size // 2:]
    return x_sel, y_sel, x_val, y_val, order1


def fault_detection(y, y_psedu):
    fault_num = np.sum(y != y_psedu)
    print("Mispredicted_inputs_num: {}".format(fault_num))

    diverse_fault_num = diverse_errors_num(y, y_psedu)
    print("Diverse_mispredicted_num: {}/{}".format(diverse_fault_num, 90))
    return fault_num, diverse_fault_num


def retrain(model_path, x, y, base_path):
    M = load_model(model_path)
    filepath = os.path.join(base_path, "temp.h5")
    trained_model = train_model(M, filepath, x,
                                keras.utils.np_utils.to_categorical(y, 10), x_val,
                                keras.utils.np_utils.to_categorical(y_val, 10))
    acc_val1 = trained_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10))[1]
    print("retrain model path: {}".format(filepath))
    print("train acc improve {} -> {}".format(acc_val0, acc_val1))
    return acc_val1


# a demo for ATS
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--m", "-m", help="Model_name", type=str, default="LeNet5"
    )
    parser.add_argument(
        "--ID",
        "-ID",
        help="The ID of running",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "Cifar10", "Fashion_mnist", "SVHN"], "Dataset should be either 'mnist' or 'cifar10'"
    assert args.m in ["LeNet1", "LeNet4", "LeNet5", "12Conv", "ResNet20"], "Model should be either 'LeNet1' or 'LeNet5'"
    
    print(args)





    # initial ATS
    base_path = "demo"
    os.makedirs(base_path, exist_ok=True)

    ats = ATS()
    data_name=args.d
    Model_name= args.m
    IDD=args.ID

    CLIP_MIN = -0.5
    CLIP_MAX = 0.5

    # mnist data
    ##
    # color_print("load LeNet-5 model and MNIST data sets", "blue")
    # print(dau)
    # (x_train, _), (x_test, y_test) = dau.load_data(use_norm=True)
    ##MNISt
    if data_name =="mnist":
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      x_train = x_train.reshape(-1, 28, 28, 1)
      x_test = x_test.reshape(-1, 28, 28, 1)
      x_train = x_train.astype("float32")
      x_test = x_test.astype("float32")
      if Model_name== "LeNet5":
        index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/ATS-master_final/Index_WN/mnist/Inw_mnist_LeNet5.npy")
        model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
      if Model_name=="LeNet1":
        index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/ATS-master_final/Index_WN/mnist/Inw_mnist_LeNet1.npy")
        model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet1)


    if data_name =="Cifar10":
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
      x_train = x_train.astype("float32")
      x_test = x_test.astype("float32")
      if Model_name=="12Conv":
        # index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/A_Paper2/ATS-master_final/Index_WN/cifar10/Inw_cifar10_12Conv.npy")
        index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/NewRQ/N9/Inw_cifar10_12Conv.npy")
        model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.Conv12)
      if Model_name=="ResNet20":
        index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/ATS-master_final/Index_WN/cifar10/Inw_cifar10_ResNet20.npy")
        model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.ResNet20)
    if data_name=="SVHN":
      train_raw = loadmat('/content/drive/MyDrive/Data/train_32x32.mat')
      test_raw = loadmat('/content/drive/MyDrive/Data/test_32x32.mat')
      x_train = np.array(train_raw['X'])
      x_test = np.array(test_raw['X'])
      y_train = train_raw['y']
      y_test = test_raw['y']
      x_train = np.moveaxis(x_train, -1, 0)
      x_test = np.moveaxis(x_test, -1, 0)
      x_test= x_test.reshape (-1,32,32,3)
      x_train= x_train.reshape (-1,32,32,3)
      x_train = x_train.astype("float32")
      x_test = x_test.astype("float32")
      lb = LabelBinarizer()
      y_train = lb.fit_transform(y_train)
      y_test = lb.fit_transform(y_test)
      y_test=np.argmax(y_test, axis=1)
      if Model_name== "LeNet5":
        index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/ATS-master_final/Index_WN/SVHN/Inw_SVHN_LeNet5.npy")
        model_path = model_conf.get_model_path(model_conf.svhn, model_conf.LeNet5)
    if data_name=="Fashion_mnist":
    # load dataset
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
      x_train = x_train.reshape(-1, 28, 28, 1)
      x_test = x_test.reshape(-1, 28, 28, 1)
      x_train = x_train.astype("float32")
      x_test = x_test.astype("float32")  
      if Model_name== "LeNet4":
        index_without_noisy=np.load("/content/drive/MyDrive/A_Paper2/ATS-master_final/Index_WN/Fashion_mnist/Inw_Fashion_mnist_LeNet4.npy")
        model_path = model_conf.get_model_path(model_conf.fashion, model_conf.LeNet4)
    
    color_print("load  model and data sets", "blue")
    ##
    if data_name!="SVHN":
      y_test = np_utils.to_categorical(y_test, 10)
      y_test=np.argmax(y_test, axis=1)  
      y_train = np_utils.to_categorical(y_train, 10)
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
    
    test_size = len(x_test)
    print("LLLoooo", test_size)
    nb_classes = model_conf.fig_nb_classes

    ori_model = load_model(model_path)

    acc = ori_model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10), verbose=0)[1]
    print("ori test accuracy {}".format(acc))

    # data augmentation
    color_print("data augmentation", "blue")
    # dau.run("test")

    ##TAGHIR
    # x_dau, y_dau = dau.load_dau_data("ori", use_cache=False)
    ##
    
    x_dau, y_dau = x_test[index_without_noisy], y_test[index_without_noisy]
    print("LLL", x_dau.shape)
    # ori_order=np.array(range(len(x_dau)))

    x_dau, y_dau, t1_order = shuffle_data3(x_dau, y_dau, index_without_noisy)

    # print("KKKKKKKKKKKKKKKKKKKKKKKKKKKK",x_dau[0]==x_test[t1_order[0]])
    
    # selection
    color_print("adaptive test selection on the augmented data", "blue")
    x_sel, y_sel, x_val, y_val , t2_order= get_tests(x_dau, y_dau, t1_order)
    acc_val0 = ori_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)[1]

    y_sel_psedu = get_psedu_label(ori_model, x_sel)
    div_rank, _, _ = ats.get_priority_sequence(x_sel, y_sel_psedu, nb_classes, ori_model, th=0.001)

    xs, ys, ys_psedu = x_sel[div_rank], y_sel[div_rank], y_sel_psedu[div_rank]
    print("priority sequence : {} ...".format(div_rank[:10]))
    print(div_rank[0])
    print(t2_order[:10])
    print(t2_order[div_rank[0]])
    #Check to see the mapping is correct
    print("KKKKKKKKKKKKKKKKKKKKKKKKKKKK",np.array_equiv(x_sel[div_rank], x_test[t2_order[div_rank]]))
    print(t2_order[div_rank])
    final_indexes=t2_order[div_rank]

    # np.save("/content/drive/MyDrive/ATS-master_final/result/X_index_cifar10_12Conv_5.npy",div_rank)
    # np.save("/content/drive/MyDrive/ATS-master_final/result/Y_cifar10_12Conv.npy",ys)
    # np.save("",ys_psedu)
    # 1000
    color_print("Select the first 1000 augmented data", "blue")
    num = 1000
    # ATS
    xs_num, ys_num, ys_psedu_num = xs[:num], ys[:num], ys_psedu[:num]
    np.save("/content/drive/MyDrive/A_Paper2/ATS-master_final/Final_subsets_baseline/" +str(data_name) +"_"+str(Model_name)+"/Index_1000_"+str(IDD)+".npy", final_indexes[:num])
    np.save("/content/drive/MyDrive/A_Paper2/ATS-master_final/Final_subsets_baseline/" +str(data_name) +"_"+str(Model_name)+"/X_1000_"+str(IDD)+".npy",xs_num)
    np.save("/content/drive/MyDrive/A_Paper2/ATS-master_final/Final_subsets_baseline/" +str(data_name) +"_"+str(Model_name)+"/Y_1000_"+str(IDD)+".npy",ys_num)
    # Random
    xr, yr, yr_psedu = shuffle_data3(x_sel, y_sel, y_sel_psedu)
    xr_num, yr_num, yr_psedu_num = xr[:num], yr[:num], yr_psedu[:num]

    # fault detection
    color_print("fault detection on selected data", "blue")
    print("ATS")
    fault_num, diverse_fault_num = fault_detection(ys_num, ys_psedu_num)
    print("Random")
    fault_num2, diverse_fault_num2 = fault_detection(yr_num, yr_psedu_num)
    color_print("Misprediction detection difference between ATS and Random: {}".format(fault_num - fault_num2), "green")
    color_print(
        "Diverse misprediction detection difference between ATS and Random: {}".format(diverse_fault_num - diverse_fault_num2),
        "green")

    # retrain
    # ATS
    # color_print("retrain model on selected data", "blue")
    # print("ATS")
    # acc_val1 = retrain(model_path, xs_num, ys_num, base_path)
    # print("Random")
    # acc_val2 = retrain(model_path, xr_num, yr_num, base_path)

    # color_print("accuracy difference between ATS and Random :{}".format(acc_val1 - acc_val2), "green")
