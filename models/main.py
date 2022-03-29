import torchvision.models as models
import torch
import torch.nn as nn
from  models.ResNetCBAMAA import *
# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnet34(pretrained=use_pretrained) #使用预训练权重
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def get_res_model(num_classes,res_layer,model_path=None):

    if(res_layer==34):

        model_ft = models.resnet34()
    elif (res_layer==50):
        model_ft = models.resnet50()
    else :
        #default 34
        model_ft = models.resnet34()

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    model_ft.load_state_dict(torch.load(model_path))
    return model_ft
def res_model_pretrain(res_layer,num_classes):
    if(res_layer==34):
        model=resnet34(pretrained=True,num_classes=num_classes)
    elif(res_layer==50):
        model = resnet50(pretrained=True, num_classes=num_classes)

    return model
def get_res_model_my(num_classes,res_layer=34,model_path=None):

    if(res_layer==34):
        model=resnet34(pretrained=False,num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
    elif(res_layer==50):
        model = resnet50(pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
    return model
if __name__ == '__main__':
    """
    test
    """
    model=res_model(3)
    print(model.parameters())
    torch.save(model.state_dict(), "D:/dataset_garb/code/weight/best.pt")