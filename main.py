from utils import *
data_path = '.data'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_scripted.pth'
scripted_quantized_model_file = 'mobilenet_scripted_quantized.pth'


def load_mobilenet_data():
    model = torch.hub.load('pytorch/vision:v0.13.0', 'mobilenet_v2', pretrained=True)
    torch.jit.save(torch.jit.script(model), saved_model_dir + float_model_file)

def float_test():
    data_loader, data_loader_test = prepare_data_loaders(data_path)
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')
    eval_batch_size = 50

    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.
    float_model.eval()
    # Fuses modules
    float_model.fuse_model()
    print(float_model)

    num_eval_batches = 1000

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 \
            {top5.global_avg:.3f}'.format(top1=top1, top5=top5))
    return

def QATtest():
    criterion = nn.CrossEntropyLoss()
    data_loader, data_loader_test = prepare_data_loaders(data_path)
    num_eval_batches = 1000
    num_train_batches = 20
    eval_batch_size = 50
    qat_model = load_model(saved_model_dir + float_model_file)

    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(qat_model, inplace=True)

    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for nepoch in range(1):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, 
            torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model,criterion, data_loader_test, 
            neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, 
            num_eval_batches * eval_batch_size, top1.avg))
    save_QATparams(quantized_model)

def save_QATparams(quantized_model): 
    f = open("./params_int.txt",'w')
    index = 0
    for layer,param in quantized_model.state_dict().items(): # param is weight or bias(Tensor)         
        print(layer, type(param))
        f.write(layer)
        f.write('\n')
        f.write(str(param.int_repr()))
        index = index + 1
        break
    f.close()

if __name__ == "__main__":
    QATtest()