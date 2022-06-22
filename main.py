import os
import numpy as np
import torch
import time
import random
import datetime
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from sup.data_provider import TB_Dataset
from sup.scheduler import GradualWarmupScheduler
from sup.contrastive_loss import con_loss, multi_con_loss
from sup.predictTCL import predictTCL

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def train_model(model, optimizer, dataload, scheduler, scheduler_warmup, num_epochs, loss_type):
    loss_ls = []
    step_ls = []

    start_time = datetime.datetime.now()
    print('epoch start time: ', start_time)

    train_info = str(loss_type) + ',   drop:' + str(dropout_value) + ',   weight_decay:' + str(
        optimizer.state_dict()['param_groups'][0]['weight_decay'])
    train_info = 'batch_size:' + str(batch_size) + '    ' + train_info

    if model_type == 'predictTCL':
        train_info = str(model_type) + '   con_loss' + str(con_loss_type) + '  lamb_sup:' + str(lamb_sup) + '  lamb_con:' + str(lamb_con) + '  tau:' + str(
            tau) + '    ' + train_info
    loss_ls.append(train_info)
    loss_ls.append(start_time)
    step_ls.append(train_info)
    step_ls.append(start_time)


    if model_type == 'predictTCL':
        weight_path = os.path.join(weight_save_path, str(str(start_time).split(' ')[0]) + '-' + str((str(start_time).split(' ')[1]).split('.')[0]).split(':')[0] + '_' +
                                   str((str(start_time).split(' ')[1]).split('.')[0]).split(':')[1], str(fold) + '_' + str(con_loss_type) + '_' + str(lamb_sup) + '_' + str(lamb_con))


    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    for epoch in range(num_epochs):

        # print('Epoch {}/{}'.format(epoch+1, num_epochs), ',  lr: ' + str(optimizer.param_groups[0]['lr']))
        print('\nEpoch {}/{}'.format(epoch + 1, num_epochs),
              ',  lr: ' + str(optimizer.state_dict()['param_groups'][0]['lr']))

        print('-' * 60)

        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for inputs, labels, patient in dataload:
            step += 1

            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            if model_type == 'predictTCL':
                con_share_inv, con_share_men, con_spec_inv, con_spec_men, sup_out_inv, sup_out_men, out_inv, out_men = model(
                    inputs)
                main_loss_inv = F.cross_entropy(out_inv, labels[:, 0])
                main_loss_men = F.cross_entropy(out_men, labels[:, 1])
                sup_loss_inv = F.cross_entropy(sup_out_inv, labels[:, 0])
                sup_loss_men = F.cross_entropy(sup_out_men, labels[:, 1])
                if con_loss_type == 'ori':
                    loss_con_inv = con_loss(con_share_inv, con_spec_inv, con_spec_men, tau, False)
                    loss_con_men = con_loss(con_share_men, con_spec_men, con_spec_inv, tau, False)
                elif con_loss_type == 'multi':
                    loss_con_inv = multi_con_loss(con_share_inv, con_spec_inv, con_spec_men, con_share_men, tau)
                    loss_con_men = multi_con_loss(con_share_men, con_spec_men, con_spec_inv, con_share_inv, tau)
                if ifcon_epoch:
                    if epoch + 1 < con_epoch:
                        loss_con_inv = 0
                        loss_con_men = 0
                loss = main_loss_inv + main_loss_men + lamb_sup * (sup_loss_inv + sup_loss_men) + lamb_con * (
                            loss_con_inv + loss_con_men)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if num_classes == 2:
                pred_inv = out_inv.argmax(dim=1, keepdim=True)
                pred_men = out_men.argmax(dim=1, keepdim=True)

            correct_inv = pred_inv.eq(labels[:, 0].view_as(pred_inv)).sum().item()
            correct_men = pred_men.eq(labels[:, 1].view_as(pred_men)).sum().item()

            display_num = int(128 / batch_size) * 5
            if step % display_num == 0:
                # invasion
                new_label_inv = labels[:, 0].view_as(pred_inv)
                right_0_inv = 0
                right_1_inv = 0
                for i in range(pred_inv.shape[0]):
                    if pred_inv[i, 0] == new_label_inv[i, 0] and pred_inv[i, 0] == 1:
                        right_1_inv += 1
                    elif pred_inv[i, 0] == new_label_inv[i, 0] and pred_inv[i, 0] == 0:
                        right_0_inv += 1
                num_0_inv = len([i for i, x in enumerate(labels[:, 0]) if x == 0])
                num_1_inv = len([i for i, x in enumerate(labels[:, 0]) if x == 1])
                # Men
                new_label_men = labels[:, 1].view_as(pred_men)
                right_0_men = 0
                right_1_men = 0
                for i in range(pred_men.shape[0]):
                    if pred_men[i, 0] == new_label_men[i, 0] and pred_men[i, 0] == 1:
                        right_1_men += 1
                    elif pred_men[i, 0] == new_label_men[i, 0] and pred_men[i, 0] == 0:
                        right_0_men += 1
                num_0_men = len([i for i, x in enumerate(labels[:, 1]) if x == 0])
                num_1_men = len([i for i, x in enumerate(labels[:, 1]) if x == 1])

                if model_type == 'predictTCL':
                    if ifcon_epoch:
                        if epoch + 1 < con_epoch:
                            print(' ' * 5 + str(step) + '/' + str(
                                (dt_size - 1) // dataload.batch_size + 1) + ' || loss: ' + str(
                                loss.item()) + ', main_loss_inv: ' + str(
                                main_loss_inv.item()) + ', loss_men: ' + str(
                                main_loss_men.item()) + ', sup_loss_inv: ' + str(
                                lamb_sup * sup_loss_inv.item()) + ', sup_loss_men: ' + str(
                                lamb_sup * sup_loss_men.item()) + ', loss_con_inv: ' + str(
                                0) + ', loss_con_men: ' + str(
                                0)
                                  + ' ||  Acc_inv: {}/{} ({:.0f}%'.format(correct_inv, batch_size,
                                                                          100. * correct_inv / batch_size) + ' ( 0: ' + str(
                                right_0_inv) + '/' + str(num_0_inv) + ', 1: ' + str(
                                right_1_inv) + '/' + str(num_1_inv)
                                  + '), Acc_men: {}/{} ({:.0f}%'.format(correct_men, batch_size,
                                                                        100. * correct_men / batch_size) + ', 0: ' + str(
                                right_0_men) + '/' + str(num_0_men) + ', 1: ' + str(
                                right_1_men) + '/' + str(num_1_men) + ')')

                            step_ls.append('epoch:' + str(epoch) + ', ' + str(step) + '/' + str(
                                (dt_size - 1) // dataload.batch_size + 1) + ' || loss: ' + str(
                                loss.item()) + ', main_loss_inv: ' + str(
                                main_loss_inv.item()) + ', main_loss_men: ' + str(
                                main_loss_men.item()) + ', sup_loss_inv: ' + str(
                                lamb_sup * sup_loss_inv.item()) + ', sup_loss_men: ' + str(
                                lamb_sup * sup_loss_men.item()) + ', loss_con_inv: ' + str(
                                0) + ', loss_con_men: ' + str(
                                0)
                                           + ' ||  Acc_inv: {}/{} ({:.0f}%'.format(correct_inv,
                                                                                   batch_size,
                                                                                   100. * correct_inv / batch_size) + ' ( 0: ' + str(
                                right_0_inv) + '/' + str(num_0_inv) + ', 1: ' + str(
                                right_1_inv) + '/' + str(num_1_inv) +
                                           '), Acc_men: {}/{} ({:.0f}%'.format(correct_men, batch_size,
                                                                               100. * correct_men / batch_size) + ', 0: ' + str(
                                right_0_men) + '/' + str(num_0_men) + ', 1: ' + str(
                                right_1_men) + '/' + str(num_1_men) + ')')
                        else:
                            print(' ' * 5 + str(step) + '/' + str(
                                (dt_size - 1) // dataload.batch_size + 1) + ' || loss: ' + str(
                                loss.item()) + ', main_loss_inv: ' + str(
                                main_loss_inv.item()) + ', loss_men: ' + str(
                                main_loss_men.item()) + ', sup_loss_inv: ' + str(
                                lamb_sup * sup_loss_inv.item()) + ', sup_loss_men: ' + str(
                                lamb_sup * sup_loss_men.item()) + ', loss_con_inv: ' + str(
                                lamb_con * loss_con_inv.item()) + ', loss_con_men: ' + str(
                                lamb_con * loss_con_men.item())
                                  + ' ||  Acc_inv: {}/{} ({:.0f}%'.format(correct_inv, batch_size,
                                                                          100. * correct_inv / batch_size) + ' ( 0: ' + str(
                                right_0_inv) + '/' + str(num_0_inv) + ', 1: ' + str(
                                right_1_inv) + '/' + str(num_1_inv)
                                  + '), Acc_men: {}/{} ({:.0f}%'.format(correct_men, batch_size,
                                                                        100. * correct_men / batch_size) + ', 0: ' + str(
                                right_0_men) + '/' + str(num_0_men) + ', 1: ' + str(
                                right_1_men) + '/' + str(num_1_men) + ')')

                            step_ls.append('epoch:' + str(epoch) + ', ' + str(step) + '/' + str(
                                (dt_size - 1) // dataload.batch_size + 1) + ' || loss: ' + str(
                                loss.item()) + ', main_loss_inv: ' + str(
                                main_loss_inv.item()) + ', main_loss_men: ' + str(
                                main_loss_men.item()) + ', sup_loss_inv: ' + str(
                                lamb_sup * sup_loss_inv.item()) + ', sup_loss_men: ' + str(
                                lamb_sup * sup_loss_men.item()) + ', loss_con_inv: ' + str(
                                lamb_con * loss_con_inv.item()) + ', loss_con_men: ' + str(
                                lamb_con * loss_con_men.item())
                                           + ' ||  Acc_inv: {}/{} ({:.0f}%'.format(correct_inv,
                                                                                   batch_size,
                                                                                   100. * correct_inv / batch_size) + ' ( 0: ' + str(
                                right_0_inv) + '/' + str(num_0_inv) + ', 1: ' + str(
                                right_1_inv) + '/' + str(num_1_inv) +
                                           '), Acc_men: {}/{} ({:.0f}%'.format(correct_men, batch_size,
                                                                               100. * correct_men / batch_size) + ', 0: ' + str(
                                right_0_men) + '/' + str(num_0_men) + ', 1: ' + str(
                                right_1_men) + '/' + str(num_1_men) + ')')
                    else:
                        print(' ' * 5 + str(step) + '/' + str(
                            (dt_size - 1) // dataload.batch_size + 1) + ' || loss: ' + str(
                            loss.item()) + ', main_loss_inv: ' + str(
                            main_loss_inv.item()) + ', loss_men: ' + str(
                            main_loss_men.item()) + ', sup_loss_inv: ' + str(
                            lamb_sup * sup_loss_inv.item()) + ', sup_loss_men: ' + str(
                            lamb_sup * sup_loss_men.item()) + ', loss_con_inv: ' + str(
                            lamb_con * loss_con_inv.item()) + ', loss_con_men: ' + str(
                            lamb_con * loss_con_men.item())
                              + ' ||  Acc_inv: {}/{} ({:.0f}%'.format(correct_inv, batch_size,
                                                                      100. * correct_inv / batch_size) + ' ( 0: ' + str(
                            right_0_inv) + '/' + str(num_0_inv) + ', 1: ' + str(
                            right_1_inv) + '/' + str(num_1_inv)
                              + '), Acc_men: {}/{} ({:.0f}%'.format(correct_men, batch_size,
                                                                    100. * correct_men / batch_size) + ', 0: ' + str(
                            right_0_men) + '/' + str(num_0_men) + ', 1: ' + str(
                            right_1_men) + '/' + str(num_1_men) + ')')

                        step_ls.append('epoch:' + str(epoch) + ', ' + str(step) + '/' + str(
                            (dt_size - 1) // dataload.batch_size + 1) + ' || loss: ' + str(
                            loss.item()) + ', main_loss_inv: ' + str(
                            main_loss_inv.item()) + ', main_loss_men: ' + str(
                            main_loss_men.item()) + ', sup_loss_inv: ' + str(
                            lamb_sup * sup_loss_inv.item()) + ', sup_loss_men: ' + str(
                            lamb_sup * sup_loss_men.item()) + ', loss_con_inv: ' + str(
                            lamb_con * loss_con_inv.item()) + ', loss_con_men: ' + str(
                            lamb_con * loss_con_men.item())
                                       + ' ||  Acc_inv: {}/{} ({:.0f}%'.format(correct_inv, batch_size,
                                                                               100. * correct_inv / batch_size) + ' ( 0: ' + str(
                            right_0_inv) + '/' + str(num_0_inv) + ', 1: ' + str(
                            right_1_inv) + '/' + str(num_1_inv) +
                                       '), Acc_men: {}/{} ({:.0f}%'.format(correct_men, batch_size,
                                                                           100. * correct_men / batch_size) + ', 0: ' + str(
                            right_0_men) + '/' + str(num_0_men) + ', 1: ' + str(
                            right_1_men) + '/' + str(num_1_men) + ')')


        print(datetime.datetime.now(), ' epoch ' + str(epoch + 1), ': loss is ' + str(epoch_loss / step))
        if (epoch + 1) % 10 == 0 and (epoch + 1) > 40:
            torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(num_epochs) + '_' + str(
                                                            batch_size) + '_epoch' + str(epoch + 1) + '.pth'))

        if ifwarmup:
            scheduler_warmup.step(epoch)
        else:
            scheduler.step()

        loss_ls.append(
            'Epoch {}/{}'.format(epoch + 1, num_epochs) + ', loss: ' + str(epoch_loss / step) + ',  lr:' + str(
                optimizer.state_dict()['param_groups'][0]['lr']))

    end_time = datetime.datetime.now()
    loss_ls.append(end_time)
    loss_ls_pd = pd.DataFrame(loss_ls)
    step_ls_pd = pd.DataFrame(step_ls)

    loss_ls_pd.to_csv(
        os.path.join(weight_path, fold + '_train_loss_' + str(num_epochs) + '_' + str(batch_size) + '.csv'),
        index=False, header=False)
    step_ls_pd.to_csv(
        os.path.join(weight_path, fold + '_train_step_' + str(num_epochs) + '_' + str(batch_size) + '.csv'),
        index=False, header=False)
    torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(num_epochs) + '_' + str(
        batch_size) + '.pth'))

def model_select():
    if model_type == 'predictTCL':
        print('predictTCL........')
        model = predictTCL(num_classes, dropout_value, run_type)
    return model

def train(batch_size, lr, epoches, data_path, loss_type):
    model = model_select()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Num of parameters: ', sum([np.prod(p.size()) for p in model_parameters]))
    model = model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    if ifwarmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    train_set = TB_Dataset('train', data_path, if_offline_data_aug, ifaddadc)

    dataloaders = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)

    if ifwarmup:
        train_model(model, optimizer, dataloaders, scheduler, scheduler_warmup, epoches, loss_type)
    else:
        train_model(model, optimizer, dataloaders, scheduler, False, epoches, loss_type)

def test(weight_path, data_path, num_classes):

    model = model_select()

    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()

    test_set = TB_Dataset('test', data_path, False, ifaddadc)

    dataloaders = DataLoader(test_set, batch_size=1, num_workers=12)

    count = 0
    correct_inv = 0
    TP_inv = 0
    FP_inv = 0
    FN_inv = 0
    TN_inv = 0

    correct_men = 0
    TP_men = 0
    FP_men = 0
    FN_men = 0
    TN_men = 0

    y_score_inv = []
    y_true_inv = []
    y_score_men = []
    y_true_men = []

    pred_invs = []  # save inv predict result
    pred_mens = []  # save men predict result

    labels = []  # save labels


    with torch.no_grad():
        for x, y, patient in dataloaders:
            count += 1
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            y = y.cuda()
            labels.append(y.cpu().numpy())
            out_inv, out_men = model(x)

            print("-----------", count, "-----------")
            print(patient)

            # invasion
            out_inv = F.softmax(out_inv, dim=1)
            pred_inv = out_inv.argmax(dim=1, keepdim=True)
            y_score_inv.append(out_inv[0, 1].cpu())

            pred_invs.append(pred_inv.cpu().numpy()[0])
            y_true_inv.append(y[:, 0].cpu())
            if num_classes == 2:
                print(out_inv[:, 1])
            print("predict_inv: {}".format(pred_inv.view_as(y[:, 0])))
            print('target_inv:  {}'.format(y[:, 0]))

            correct_inv += pred_inv.eq(y[:, 0].view_as(pred_inv)).sum().item()
            if pred_inv.view_as(y[:, 0]) == y[:, 0] == 1: TP_inv += 1
            if (pred_inv.view_as(y[:, 0]) == 1) and (y[:, 0] == 0): FP_inv += 1
            if (pred_inv.view_as(y[:, 0]) == 0) and (y[:, 0] == 1): FN_inv += 1
            if pred_inv.view_as(y[:, 0]) == y[:, 0] == 0: TN_inv += 1


            # meningioma
            out_men = F.softmax(out_men, dim=1)
            pred_men = out_men.argmax(dim=1, keepdim=True)
            y_score_men.append(out_men[0, 1].cpu())

            pred_mens.append(pred_men.cpu().numpy()[0])
            y_true_men.append(y[:, 1].cpu())
            if num_classes == 2:
                print(out_men[:, 1])
            else:
                print(out_men)
            print("predict_men: {}".format(pred_men.view_as(y[:, 1])))
            print('target_men:  {}'.format(y[:, 1]))

            correct_men += pred_men.eq(y[:, 1].view_as(pred_men)).sum().item()
            if pred_men.view_as(y[:, 1]) == y[:, 1] == 1: TP_men += 1
            if (pred_men.view_as(y[:, 1]) == 1) and (y[:, 1] == 0): FP_men += 1
            if (pred_men.view_as(y[:, 1]) == 0) and (y[:, 1] == 1): FN_men += 1
            if pred_men.view_as(y[:, 1]) == y[:, 1] == 0: TN_men += 1

        end_time = datetime.datetime.now()
        print(end_time)

        # Invasion
        fpr_inv, tpr_inv, thresholds_inv = roc_curve(y_true_inv, y_score_inv, pos_label=None,
                                                     sample_weight=None, drop_intermediate=True)
        roc_auc_inv = auc(fpr_inv, tpr_inv)
        labels = np.array(labels)
        labels = np.squeeze(labels, axis=1)
        balance_acc_inv = balanced_accuracy_score(labels[:, 0], pred_invs)
        mcc_inv = matthews_corrcoef(labels[:, 0], pred_invs)
        auprc_inv = average_precision_score(labels[:, 0], pred_invs)

        print('\nAcc_inv: {}/{} ({:.0f}%)'.format(correct_inv, count, 100. * correct_inv / count))
        print("TP,FP,FN,TN:", TP_inv, FP_inv, FN_inv, TN_inv)
        print('sensitivity:', TP_inv / (TP_inv + FN_inv))
        print('specificity:', TN_inv / (TN_inv + FP_inv))
        print('accuracy:', (TP_inv + TN_inv) / (TP_inv + FP_inv + FN_inv + TN_inv))
        print('g_mean:', ((TP_inv / (TP_inv + FN_inv)) * (TN_inv / (TN_inv + FP_inv))) ** 0.5)
        print('balance_acc:', str(balance_acc_inv))
        print('mcc:', str(mcc_inv))
        print('auprc:', str(auprc_inv))
        print('roc_auc', roc_auc_inv)

        # meningioma
        fpr_men, tpr_men, thresholds_men = roc_curve(y_true_men, y_score_men, pos_label=None,
                                                     sample_weight=None, drop_intermediate=True)
        roc_auc_men = auc(fpr_men, tpr_men)
        balance_acc_men = balanced_accuracy_score(pred_mens, labels[:, 1])
        mcc_men = matthews_corrcoef(labels[:, 1], pred_mens)
        auprc_men = average_precision_score(labels[:, 1], pred_mens)
        print('\nAcc_men: {}/{} ({:.0f}%)'.format(correct_men, count, 100. * correct_men / count))
        print("TP,FP,FN,TN:", TP_men, FP_men, FN_men, TN_men)
        print('sensitivity:', TP_men / (TP_men + FN_men))
        print('specificity:', TN_men / (TN_men + FP_men))
        print('accuracy:', (TP_men + TN_men) / (TP_men + FP_men + FN_men + TN_men))
        print('g_mean:', ((TP_men / (TP_men + FN_men)) * (TN_men / (TN_men + FP_men))) ** 0.5)
        print('balance_acc:', str(balance_acc_men))
        print('mcc:', str(mcc_men))
        print('auprc:', str(auprc_men))
        print('roc_auc', roc_auc_men)


if __name__ == '__main__':

    batch_size = 32
    train_lr = 0.001
    train_epochs = 1

    global weight_decay_value, fold, dropout_value, loss_typ, ifaddadc, if_offline_data_aug, ifwarmup
    global ifdrop, tau, image_type, num_classes, lamb_con, con_loss_type, run_type, lamb_sup, ifResNet_newmiddle
    global model_type, modaility_num, ifcon_epoch, con_epoch, weight_save_path

    num_classes = 2 # class number of each task
    image_type = 'bbox'
    weight_decay_value = 0.001
    if_offline_data_aug = True  # if offline data enhancement has been done
    ifwarmup = True  # train with warmup
    dropout_value = 0.5
    loss_type = 'softmax'
    ifaddadc = True
    model_type = 'predictTCL'
    ifcon_epoch = True
    weight_save_path = '/home/Invasion/weights'

    if ifcon_epoch:
        con_epoch = 30 # add contrastive loss in epoch 30
    lamb_con = 1 # weight of contrastive loss
    lamb_sup = 0.7 # weight of aux branch CE loss
    tau = 0.07
    con_loss_type = 'ori'  # ori, multi
    if ifaddadc:
        modaility_num = 3

    data_path = '/home/Invasion/data/BBox/split/random3folder/' # data path
    fold = 'random_2'
    use_path = os.path.join(data_path, fold)

    run_type = 'test'  # train or test

    print('start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if run_type == 'train':
        print(run_type + ' <<<   model_type:' + str(model_type) + '   loss_type: ' + str(
            loss_type) + '   epoch: ' + str(train_epochs) + '   lr: ' + str(train_lr) + '   batch_size: ' + str(
            batch_size) + '   weight_decay:' + str(weight_decay_value) + '   drop:' + str(dropout_value))
        if model_type == 'predictTCL':
            train(batch_size, train_lr, train_epochs, use_path, loss_type)
    elif run_type == 'test':  # test
        weight_date = '2022-03-06-10_27'  # weight_date
        fold = 'random_2'
        model_type = 'predictTCL'

        if model_type == 'predictTCL':
            weight_saved_path = os.path.join(weight_save_path, str(weight_date), fold + '_' + str(con_loss_type) + '_' + str(lamb_sup) + '_' + str(lamb_con), fold + '_train_weight_' + str(train_epochs) + '_' + str(
                batch_size) + '.pth')
            test(weight_saved_path, use_path, num_classes)

    print('\nend time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

