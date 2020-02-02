-------------------------------------------------------------------------------
-- Input arguments and options
-- 输入图片和VGG模型，输出图片的特征
-- 模块:
-- 1. 获得模型
-- 2. 图像预处理模块
-- 3. 模型处理图像模块
-- 4. 正式处理
-------------------------------------------------------------------------------
require 'nn'
require 'optim'
require 'torch'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson')
require 'xlua'

-- 命令行参数作为全局变量
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('--input_json','../data/cocoqa_data_prepro.json','path to the json file containing vocab and answers')
cmd:option('--image_root','/home/bigface/data/','path to the image root')
cmd:option('--cnn_proto', '../image_model/VGG_ILSVRC_19_layers_deploy.prototxt', 'path to the cnn prototxt')
cmd:option('--cnn_model', '../image_model/VGG_ILSVRC_19_layers.caffemodel', 'path to the cnn model')

cmd:option('--batch_size', 20, 'batch_size')

cmd:option('--out_name_train', '../data/cocoqa_data_img_vgg_train.h5', 'output name train')
cmd:option('--out_name_test', '../data/cocoqa_data_img_vgg_test.h5', 'output name test')
cmd:option('--out_name_val', '../data/cocoqa_data_img_vgg_val.h5', 'output name val')


cmd:option('--gpuid', 6, 'which gpu to use. -1 = use CPU')
cmd:option('--backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)


function getVGGNet()
    --[[
        1.载入
        2. 处理处理VGG网络：删除网络的38~46层
        ------------------------------------------
        Args:

        Returns:

    ]]
    local net = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.backend)
    for i = 46, 38, -1 do
        net:remove(i)
    end

    return net
end

function loadImage(image_path)
    --[[
        1. 载入
        2. 图像预处理：
            1) 将图像拉伸为 448 × 448，（从中心看224×224）
            2) 处理图像的通道数：如果通道数为1（黑白图像），则将图像复制三遍；如果通道数大于3，则取前三个通道
            3) 将图像逆转为BGR，并在各个通道上减去经验均值*
        ------------------------------------------
        Args:
            image_path: 图像的路径
        Returns:
            im2: image tensor, size = (3, 448, 448)
    ]]
    local im = image.load(image_path, 3, "float")
    im = image.scale(im, 448, 448)
    local dim= im:size(1)

    if dim == 1 then
        local im2 = torch.cat(im, im, 1)
        im2 = torch.cat(im2, im, 1)
        im = im2
    elseif dim == 4 then
        im = im[{{1,3}, {}, {}}]
    end

    im = im * 255
    local im2 = im:clone()
    im2[{{3}, {}, {}}] = im[{{1}, {}, {}}] - 123.68
    im2[{{2}, {}, {}}] = im[{{2}, {}, {}}] - 116.779
    im2[{{1}, {}, {}}] = im[{{3}, {}, {}}] - 103.939
    return im2
end

function batch_process(net, imgnames)
    --[[
        输入一批图像（名称），将其转化为特征张量

        ------------------------------------------
        Args:
            net: 用于处理图像的vgg网络
            imgnames: list, 包含了所有要提取特征的图像文件名

        Returns:

    ]]
    local batch_size = opt.batch_size
    local sz = #imgnames
    local feat = torch.FloatTensor(sz, 14, 14, 512)
    print(string.format('processing %d images', sz))
    for i = 1, sz, batch_size do
        xlua.progress(i, sz)
        local ub = math.min(sz, i + batch_size - 1)
        local ims = torch.CudaTensor(ub-i+1, 3, 448, 448)
        for j = 1, ub-i+1 do
            ims[j] = loadImage(opt.image_root..imgnames[i+j-1]):cuda()
        end
        net:forward(ims)
        feat[{{i, ub}, {}}] = net.modules[37].output:permute(1,3,4,2):contiguous():float() -- 通道放在最后
        collectgarbage()
    end
    return feat
end

vggnet = getVGGNet()
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()

json_file = cjson.decode(text)
train_feats = batch_process(vggnet, json_file["unique_img_train"])
test_feats = batch_process(vggnet, json_file["unique_img_test"])
val_feats = batch_process(vggnet, json_file["unique_img_val"])

local train_h5_file = hdf5.open(opt.out_name_train, 'w')
train_h5_file:write('images', train_feats)
train_h5_file:close()

local test_h5_file = hdf5.open(opt.out_name_test, 'w')
test_h5_file:write('images', test_feats)
test_h5_file:close()

local val_h5_file = hdf5.open(opt.out_name_val, 'w')
val_h5_file:write('images', val_feats)
val_h5_file:close()

