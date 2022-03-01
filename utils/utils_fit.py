import torch
from tqdm import tqdm
import sys
from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,save_model=False):
    '''
    :param model_train:  yolobody.train方法
    :param model:   yolobody类
    :param yolo_loss: yololoss类
    :param loss_history: 损失计算类
    :param optimizer: 优化器对象
    :param epoch: 当前进行的epoch索引,表示数据集训练了多少遍
    :param epoch_step:  训练数据量/batch 每训练完一遍数据集会停下来保存数据
    :param epoch_step_val:  验证数据量/batch
    :param gen:
    :param gen_val:
    :param Epoch: 结束的epoch索引
    :param cuda:
    :return:
    '''
    loss        = 0
    val_loss    = 0

    #TODO: model.train()方法是否可以多次调用
    model_train.train()
    # print('\nStart Train an epoch\n')
    # python进度条
    with tqdm(total=epoch_step,desc=f'Train Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=2.0) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            num_pos_all     = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                # output返回的是三张特征图 对应分别为13x13x75 26x26x75 52x52x75的特征图 的tensor组成的元组
                # target对应每张图片的obj
                # 第一个维度 batch
                # 第二个维度 每张图片的目标数量
                # 第三个维度 左上角xy和右下角xy 置信度
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                # num_pos 每个batch有的目标总数
                # loss_item 总共的损失,比如类别,位置等的损失
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            loss_value = loss_value_all / num_pos_all

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    model_train.eval()
    with tqdm(total=epoch_step_val, desc=f'Val Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('End a Step epoch ', epoch)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if save_model:
        loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
