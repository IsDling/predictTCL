import os
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from sup.online_aug import online_aug


def Standardize(images):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    new: z-score is used but keep the background with zero!
    """
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)
    mask_location = images.sum(0) > 0
    for k in range(images.shape[0]):
        image = images[k,...]
        image = np.array(image, dtype = 'float32')
        mask_area = image[mask_location]
        image[mask_location] -= mask_area.mean()
        image[mask_location] /= mask_area.std()
        images[k,...] = image
    return images

def multi_model_Standard(image):
    for i in range(image.shape[0]):
        img = image[i, ...]
        new_img = Standardize(img)
        if i == 0:
            out_image = new_img
        else:
            out_image = np.concatenate((out_image, new_img), axis=0)
    out_image = out_image.copy()
    return out_image

def get_image(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose((1, 2, 0)).astype(np.float)
    img = np.clip(img, 0, 1200)
    img = np.expand_dims(img, axis=0)
    return img


def load_data(t1_path, t2_path, adc_path, label, patient, ifaddadc, subset):

    t1_img = get_image(t1_path)
    t2_img = get_image(t2_path)

    input_img = np.concatenate((t1_img, t2_img), axis=0)
    if ifaddadc:
        adc_img = get_image(adc_path)
        input_img = np.concatenate((input_img, adc_img), axis=0)
    if subset == 'train':
        img_out = online_aug.online_aug(input_img, 'bbox')
    else:
        img_out = multi_model_Standard(input_img)


    label = np.array(label)

    return img_out, label, patient


def calc_label(grade_path_name, labels, num_0_inv, num_1_inv, num_0_men, num_1_men):
    if grade_path_name.split('_')[1] == '1':
        label = []
        label.append(0)
        label.append(0)
        labels.append(label)
        num_0_inv += 1
        num_0_men += 1

    if grade_path_name.split('_')[1] == '2' and grade_path_name.split('_')[-1] == 'invasion':
        label = []
        label.append(1)
        label.append(1)
        labels.append(label)
        num_1_inv += 1
        num_1_men += 1

    if grade_path_name.split('_')[1] == '2' and grade_path_name.split('_')[-1] == 'noninvasion':
        label = []
        label.append(0)
        label.append(1)
        labels.append(label)
        num_0_inv += 1
        num_1_men += 1
    return labels, num_0_inv, num_1_inv, num_0_men, num_1_men


def calc_label_multi(grade_path_name, labels, num_0, num_1, num_2):
    if grade_path_name.split('_')[1] == '1':
        labels.append(0)
        num_0 += 1

    if grade_path_name.split('_')[1] == '2' and grade_path_name.split('_')[-1] == 'invasion':
        labels.append(1)
        num_1 += 1

    if grade_path_name.split('_')[1] == '2' and grade_path_name.split('_')[-1] == 'noninvasion':
        labels.append(2)
        num_2 += 1
    return labels, num_0, num_1, num_2


class TB_Dataset(Dataset):

    def __init__(self, subset, data_path, if_offline_data_aug, ifaddadc):
        super(TB_Dataset, self).__init__()
        self.ifaddadc = ifaddadc
        self.subset = subset

        # dataset path
        if 'train' in subset:
            source_path = os.path.join(data_path, 'train')
        else:
            source_path = os.path.join(data_path, subset)
        print(source_path)

        patients = []
        labels = []
        t1 = []
        t2 = []
        adc = []
        modal = []

        num_1_inv = 0
        num_0_inv = 0
        num_1_men = 0
        num_0_men = 0

        for grade_path_name in os.listdir(source_path):
            grade_path = os.path.join(source_path, grade_path_name)
            for patient_path_name in os.listdir(grade_path):
                patient_path = os.path.join(grade_path, patient_path_name)
                if if_offline_data_aug:
                    for nii_path_name in os.listdir(patient_path):
                        if nii_path_name.startswith('t1_bbox'):
                            t1_path = os.path.join(patient_path, nii_path_name)
                            t1.append(t1_path)
                            patients.append(patient_path_name)
                            t2_path = os.path.join(patient_path, 't2_bbox'+str(nii_path_name.split('t1_bbox')[1]))
                            if os.path.exists(t2_path):
                                t2.append(t2_path)
                            else:
                                print('not exist ' + t2_path)
                            if ifaddadc:
                                adc_path = os.path.join(patient_path,'adc_bbox' + str(nii_path_name.split('t1_bbox')[1]))
                                if os.path.exists(adc_path):
                                    adc.append(adc_path)
                                else:
                                    print('not exist ' + adc_path)
                            labels, num_0_inv, num_1_inv, num_0_men, num_1_men = calc_label(grade_path_name, labels, num_0_inv, num_1_inv, num_0_men, num_1_men)
                else:
                    t1_path = os.path.join(patient_path, 't1_bbox.nii.gz')
                    if os.path.exists(t1_path):
                        t1.append(t1_path)
                        patients.append(patient_path_name)

                        t2_path = os.path.join(patient_path, 't2_bbox.nii.gz')
                        if os.path.exists(t2_path):
                            t2.append(t2_path)
                        else:
                            print('not exist ' + t2_path)

                        if ifaddadc:
                            adc_path = os.path.join(patient_path, 'adc_bbox.nii.gz')
                            if os.path.exists(adc_path):
                                adc.append(adc_path)
                            else:
                                print('not exist ' + adc_path)
                        labels, num_0_inv, num_1_inv, num_0_men, num_1_men = calc_label(grade_path_name, labels, num_0_inv, num_1_inv, num_0_men, num_1_men)
                    else:
                        print('not exist ' + t1_path)

        print('Num of all samples:', len(labels))
        print('Invasion || Num of label 0:', num_0_inv, '     Num of label 1:', num_1_inv)
        print('meningioma || Num of label 0:', num_0_men,'     Num of label 1:', num_1_men)

        self.t1 = t1
        self.t2 = t2
        self.adc = adc
        self.modal = modal
        self.labels = labels
        self.patients = patients


    def __getitem__(self, index):

        if self.ifaddadc:
            x1 = self.t1[index]
            x2 = self.t2[index]
            x3 = self.adc[index]
        img_out, label, patient = load_data(x1, x2, x3, self.labels[index], self.patients[index], self.ifaddadc, self.subset)
        return img_out, label, patient

    def __len__(self):
        return len(self.labels)
