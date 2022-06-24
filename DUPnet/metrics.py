import numpy as np
import os
import cv2

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        meanAcc_class = np.nanmean(Acc)
        return meanAcc_class
        # 精确率(Precision)

    def pixelPrecision(self):
        # return all class overall pixel Precision
        # Precision = TP  / (TP  + FP )
        Preci = self.confusion_matrix[0][0] / np.sum(self.confusion_matrix, axis=1)[0]
        return Preci

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = np.maximum(np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix), 1)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    # Recall(召回率)
    def Pixelrecall(self):
        #R=TP/((TP+FN))
        reca11 = self.confusion_matrix[0][0] /np.sum(self.confusion_matrix,axis=0)[0]
        return reca11

    #F-Measure//F1
    def PixelF1(self):
        #R=2*P*R/(P+R)
        F1_P = self.confusion_matrix[0][0] / np.sum(self.confusion_matrix,axis=1)[0]
        F1_R = self.confusion_matrix[0][0] /np.sum(self.confusion_matrix,axis=0)[0]
        F1 = (2 * F1_P*F1_R)  / (F1_P + F1_R)
        return F1


    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def addBatch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def evaluate1(pre_path, label_path):
    acc_list = []
    Preci_list = []
    meanAcc_list = []
    recall_list = []
    F1_list = []
    mIoU_list = []
    fwIoU_list = []

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)
    for i, p in enumerate(pre_imgs):
        imgPredict = cv2.imread(pre_path + p)
        imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        imgPredict = np.array(imgPredict)

        imgLabel = cv2.imread(label_path + lab_imgs[i])
        imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        imgLabel = np.array(imgLabel)



        metric = Evaluator(2)  # 表示分类个数，包括背景
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.Pixel_Accuracy()
        Preci = metric.pixelPrecision()
        meanAcc = metric.Pixel_Accuracy_Class()
        recall = metric.Pixelrecall()
        F1 = metric.PixelF1()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()

        acc_list.append(acc)
        Preci_list.append(Preci)
        meanAcc_list.append(meanAcc)
        recall_list.append(recall)
        F1_list.append(F1)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)


    return acc_list,Preci_list, meanAcc_list, recall_list,F1_list,mIoU_list, fwIoU_list


def evaluate2(pre_path, label_path):

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)


    metric = Evaluator(2)  # 表示分类个数，包括背景

    for i, p in enumerate(pre_imgs):
        imgPredict = cv2.imread(pre_path + p)
        imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        imgPredict = np.array(imgPredict)

        imgLabel = cv2.imread(label_path + lab_imgs[i])
        imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        imgLabel = np.array(imgLabel)

        metric.addBatch(imgPredict, imgLabel)

    return metric





if __name__ == '__main__':

    pre_path ="D:\\desket\\pre\\"
    label_path ="D:\\desket\\label\\"

    # 计算测试集每张图片的各种评价指标，最后求平均
    acc_list,Preci_list,meanAcc_list, recall_list, F1_list,mIoU_list,fwIoU_list = evaluate1(pre_path, label_path)
    print('final1: acc={:.2f}%, Preci={:.2f}%,meanAcc={:.2f}%,recall={:.2f}%,F1={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%'
          .format(np.mean(acc_list) * 100,np.mean(Preci_list) * 100,np.mean(meanAcc_list) * 100,
                  np.mean(recall_list) * 100,np.mean(F1_list) * 100, np.mean(mIoU_list) * 100,np.mean(fwIoU_list) * 100))

    # 加总测试集每张图片的混淆矩阵，对最终形成的这一个矩阵计算各种评价指标
    metric = evaluate2(pre_path, label_path)
    acc = metric.Pixel_Accuracy()
    Preci = metric.pixelPrecision()
    meanAcc = metric.Pixel_Accuracy_Class()
    recall = metric.Pixelrecall()
    F1 = metric.PixelF1()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print('final2: acc={:.2f}%, Preci={:.2f}%,meanAcc={:.2f}%,recall={:.2f}%,F1={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%'
          .format(acc * 100,  Preci * 100,meanAcc * 100,recall* 100,F1 * 100, mIoU * 100, fwIoU * 100))


