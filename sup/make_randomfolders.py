import os
import random
import shutil

def make_folds_list(ori_path, num_1, num_2inv, num_2noninv):
    g1_list = []
    g2inv_list = []
    g2noninv_list = []
    label_fold = False
    if label_fold:
        # make g1_list
        for pn_path in os.listdir(os.path.join(ori_path,'label','Grade_1')):
            g1_list.append(os.path.join(ori_path,'label','Grade_1',pn_path))
        for pn_path in os.listdir(os.path.join(ori_path,'unlabel','Grade_1')):
            g1_list.append(os.path.join(ori_path, 'unlabel', 'Grade_1', pn_path))
        # make g2inv_list
        for pn_path in os.listdir(os.path.join(ori_path, 'label', 'Grade_2_invasion')):
            g2inv_list.append(os.path.join(ori_path, 'label', 'Grade_2_invasion', pn_path))
        for pn_path in os.listdir(os.path.join(ori_path, 'unlabel', 'Grade_2_invasion')):
            g2inv_list.append(os.path.join(ori_path, 'unlabel', 'Grade_2_invasion', pn_path))
        # make g2noninv_list
        for pn_path in os.listdir(os.path.join(ori_path, 'label', 'Grade_2_noninvasion')):
            g2noninv_list.append(os.path.join(ori_path, 'label', 'Grade_2_noninvasion', pn_path))
        for pn_path in os.listdir(os.path.join(ori_path, 'unlabel', 'Grade_2_noninvasion')):
            g2noninv_list.append(os.path.join(ori_path, 'unlabel', 'Grade_2_noninvasion', pn_path))
    else:
        # make g1_list
        for pn_path in os.listdir(os.path.join(ori_path, 'Grade_1')):
            g1_list.append(os.path.join(ori_path, 'Grade_1', pn_path))
        # make g2inv_list
        for pn_path in os.listdir(os.path.join(ori_path, 'Grade_2_invasion')):
            g2inv_list.append(os.path.join(ori_path, 'Grade_2_invasion', pn_path))
        # make g2noninv_list
        for pn_path in os.listdir(os.path.join(ori_path, 'Grade_2_noninvasion')):
            g2noninv_list.append(os.path.join(ori_path, 'Grade_2_noninvasion', pn_path))

    random.shuffle(g1_list)
    random.shuffle(g2inv_list)
    random.shuffle(g2noninv_list)

    one_fold_train = g1_list[:num_1]+g2inv_list[:num_2inv]+g2noninv_list[:num_2noninv]
    one_fold_test = g1_list[num_1:]+g2inv_list[num_2inv:]+g2noninv_list[num_2noninv:]

    random.shuffle(g1_list)
    random.shuffle(g2inv_list)
    random.shuffle(g2noninv_list)
    two_fold_train = g1_list[:num_1] + g2inv_list[:num_2inv] + g2noninv_list[:num_2noninv]
    two_fold_test = g1_list[num_1:] + g2inv_list[num_2inv:] + g2noninv_list[num_2noninv:]

    random.shuffle(g1_list)
    random.shuffle(g2inv_list)
    random.shuffle(g2noninv_list)
    three_fold_train = g1_list[:num_1] + g2inv_list[:num_2inv] + g2noninv_list[:num_2noninv]
    three_fold_test = g1_list[num_1:] + g2inv_list[num_2inv:] + g2noninv_list[num_2noninv:]

    return one_fold_train, one_fold_test, two_fold_train, two_fold_test, three_fold_train, three_fold_test

def copy_date_bbox(train_list, test_list, folder_path):
    print('*'*20+'make '+folder_path.split('/')[-1]+'*'*20)
    for pn_path in train_list:
        new_pn_path = os.path.join(folder_path,'train',pn_path.split('/')[-2],pn_path.split('/')[-1])
        if not os.path.exists(new_pn_path):
            os.makedirs(new_pn_path)
        shutil.copy(os.path.join(pn_path, 't1_bbox.nii.gz'), new_pn_path)
        shutil.copy(os.path.join(pn_path, 't2_bbox.nii.gz'), new_pn_path)
        shutil.copy(os.path.join(pn_path, 'adc_bbox.nii.gz'), new_pn_path)
        print('copy '+os.path.join(pn_path, 't1_bbox.nii.gz and t2_bbox.nii.gz and adc_bbox.nii.gz')+' to '+new_pn_path)
    for pn_path in test_list:
        new_pn_path = os.path.join(folder_path,'test',pn_path.split('/')[-2],pn_path.split('/')[-1])
        if not os.path.exists(new_pn_path):
            os.makedirs(new_pn_path)
        shutil.copy(os.path.join(pn_path, 't1_bbox.nii.gz'), new_pn_path)
        shutil.copy(os.path.join(pn_path, 't2_bbox.nii.gz'), new_pn_path)
        shutil.copy(os.path.join(pn_path, 'adc_bbox.nii.gz'), new_pn_path)
        print('copy '+os.path.join(pn_path, 't1_bbox.nii.gz and t2_bbox.nii.gz and adc_bbox.nii.gz')+' to '+new_pn_path)

def make_data(path, new_path, num_1, num_2inv, num_2noninv):
    one_fold_train, one_fold_test, two_fold_train, two_fold_test, three_fold_train, three_fold_test = make_folds_list(path, num_1, num_2inv, num_2noninv)
    # make random three_folder file
    one_fold_path = os.path.join(new_path,'random_1')
    copy_date_bbox(one_fold_train,one_fold_test,one_fold_path)
    two_fold_path = os.path.join(new_path, 'random_1')
    copy_date_bbox(two_fold_train,two_fold_test,two_fold_path)
    three_fold_path = os.path.join(new_path, 'random_3')
    copy_date_bbox(three_fold_train,three_fold_test,three_fold_path)

num_1 = 145
num_2inv = 44
num_2noninv = 25

make_data('/home/Invasion/data/BBox/original/','/home/Invasion/data/BBox/split/random3folder/', num_1, num_2inv, num_2noninv)






