# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.proto.framework_pb2 import VarType

import solver
from utils.config import cfg
from loss import multi_softmax_with_loss
from loss import multi_dice_loss
from loss import multi_bce_loss
from models.modeling import deeplab, unet, icnet, pspnet, hrnet, fast_scnn


class ModelPhase(object):
    """
    Standard name for model phase in PaddleSeg

    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `EVAL`: testing/evaluation mode.
    * `PREDICT`: prediction/inference mode.
    * `VISUAL` : visualization mode
    """

    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'
    VISUAL = 'visual'

    @staticmethod
    def is_train(phase):
        return phase == ModelPhase.TRAIN

    @staticmethod
    def is_predict(phase):
        return phase == ModelPhase.PREDICT

    @staticmethod
    def is_eval(phase):
        return phase == ModelPhase.EVAL

    @staticmethod
    def is_visual(phase):
        return phase == ModelPhase.VISUAL

    @staticmethod
    def is_valid_phase(phase):
        """ Check valid phase """
        if ModelPhase.is_train(phase) or ModelPhase.is_predict(phase) \
                or ModelPhase.is_eval(phase) or ModelPhase.is_visual(phase):
            return True

        return False


def seg_model(image, class_num):
    model_name = cfg.MODEL.MODEL_NAME
    if model_name == 'unet':
        logits = unet.unet(image, class_num)
    elif model_name == 'deeplabv3p':
        logits = deeplab.deeplabv3p(image, class_num)
    elif model_name == 'icnet':
        logits = icnet.icnet(image, class_num)
    elif model_name == 'pspnet':
        logits = pspnet.pspnet(image, class_num)
    elif model_name == 'hrnet':
        logits = hrnet.hrnet(image, class_num)
    elif model_name == 'fast_scnn':
        logits = fast_scnn.fast_scnn(image, class_num)
    else:
        raise Exception(
            "unknow model name, only support unet, deeplabv3p, icnet, pspnet, hrnet"
        )
    return logits


def softmax(logit):
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.softmax(logit)
    logit = fluid.layers.transpose(logit, [0, 3, 1, 2])
    return logit


def sigmoid_to_softmax(logit):
    """
    one channel to two channel
    """
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.sigmoid(logit)
    logit_back = 1 - logit
    logit = fluid.layers.concat([logit_back, logit], axis=-1)
    logit = fluid.layers.transpose(logit, [0, 3, 1, 2])
    return logit


def export_preprocess(image):
    """??????????????????????????????"""

    image = fluid.layers.transpose(image, [0, 3, 1, 2])
    origin_shape = fluid.layers.shape(image)[-2:]

    # ??????AUG_METHOD?????????resize
    if cfg.AUG.AUG_METHOD == 'unpadding':
        h_fix = cfg.AUG.FIX_RESIZE_SIZE[1]
        w_fix = cfg.AUG.FIX_RESIZE_SIZE[0]
        image = fluid.layers.resize_bilinear(
            image, out_shape=[h_fix, w_fix], align_corners=False, align_mode=0)
    elif cfg.AUG.AUG_METHOD == 'rangescaling':
        size = cfg.AUG.INF_RESIZE_VALUE
        value = fluid.layers.reduce_max(origin_shape)
        scale = float(size) / value.astype('float32')
        image = fluid.layers.resize_bilinear(
            image, scale=scale, align_corners=False, align_mode=0)

    # ??????resize?????????shape
    valid_shape = fluid.layers.shape(image)[-2:]

    # padding???eval_crop_size??????
    width = cfg.EVAL_CROP_SIZE[0]
    height = cfg.EVAL_CROP_SIZE[1]
    pad_target = fluid.layers.assign(
        np.array([height, width]).astype('float32'))
    up = fluid.layers.assign(np.array([0]).astype('float32'))
    down = pad_target[0] - valid_shape[0]
    left = up
    right = pad_target[1] - valid_shape[1]
    paddings = fluid.layers.concat([up, down, left, right])
    paddings = fluid.layers.cast(paddings, 'int32')
    image = fluid.layers.pad2d(image, paddings=paddings, pad_value=127.5)

    # normalize
    mean = np.array(cfg.MEAN).reshape(1, len(cfg.MEAN), 1, 1)
    mean = fluid.layers.assign(mean.astype('float32'))
    std = np.array(cfg.STD).reshape(1, len(cfg.STD), 1, 1)
    std = fluid.layers.assign(std.astype('float32'))
    image = (image / 255 - mean) / std
    # ?????????????????????????????????image.shape??????????????????shape
    image = fluid.layers.reshape(
        image, shape=[-1, cfg.DATASET.DATA_DIM, height, width])
    return image, valid_shape, origin_shape


def build_model(main_prog=None,
                start_prog=None,
                phase=ModelPhase.TRAIN,
                **kwargs):

    if not ModelPhase.is_valid_phase(phase):
        raise ValueError("ModelPhase {} is not valid!".format(phase))
    if ModelPhase.is_train(phase):
        width = cfg.TRAIN_CROP_SIZE[0]
        height = cfg.TRAIN_CROP_SIZE[1]
    else:
        width = cfg.EVAL_CROP_SIZE[0]
        height = cfg.EVAL_CROP_SIZE[1]

    image_shape = [-1, cfg.DATASET.DATA_DIM, height, width]
    grt_shape = [-1, 1, height, width]
    class_num = cfg.DATASET.NUM_CLASSES

    #with fluid.program_guard(main_prog, start_prog):
    #    with fluid.unique_name.guard():
    # ?????????????????????????????????????????????????????????,??????????????????????????????????????????
    # ??????????????????????????????????????????batch_size????????????
    if cfg.SLIM.KNOWLEDGE_DISTILL_IS_TEACHER:
        image = main_prog.global_block()._clone_variable(
            kwargs['image'], force_persistable=False)
        label = main_prog.global_block()._clone_variable(
            kwargs['label'], force_persistable=False)
        mask = main_prog.global_block()._clone_variable(
            kwargs['mask'], force_persistable=False)
    else:
        if ModelPhase.is_predict(phase):
            origin_image = fluid.data(
                name='image',
                shape=[-1, -1, -1, cfg.DATASET.DATA_DIM],
                dtype='float32')
            image, valid_shape, origin_shape = export_preprocess(origin_image)

        else:
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
        label = fluid.data(name='label', shape=grt_shape, dtype='int32')
        mask = fluid.data(name='mask', shape=grt_shape, dtype='int32')

    # use DataLoader.from_generator when doing traning and evaluation
    if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
        data_loader = None
        if not cfg.SLIM.KNOWLEDGE_DISTILL_IS_TEACHER:
            data_loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label, mask],
                capacity=cfg.DATALOADER.BUF_SIZE,
                iterable=False,
                use_double_buffer=True)

    loss_type = cfg.SOLVER.LOSS
    if not isinstance(loss_type, list):
        loss_type = list(loss_type)

    # dice_loss???bce_loss????????????????????????
    if class_num > 2 and (("dice_loss" in loss_type) or
                          ("bce_loss" in loss_type)):
        raise Exception(
            "dice loss and bce loss is only applicable to binary classfication")

    # ??????????????????????????????loss????????????dice_loss???bce_loss??????????????????logit????????????????????????1
    if ("dice_loss" in loss_type) or ("bce_loss" in loss_type):
        class_num = 1
        if "softmax_loss" in loss_type:
            raise Exception(
                "softmax loss can not combine with dice loss or bce loss")
    logits = seg_model(image, class_num)

    # ???????????????loss?????????????????????????????????
    if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
        loss_valid = False
        avg_loss_list = []
        valid_loss = []
        if "softmax_loss" in loss_type:
            weight = cfg.SOLVER.CROSS_ENTROPY_WEIGHT
            avg_loss_list.append(
                multi_softmax_with_loss(logits, label, mask, class_num, weight))
            loss_valid = True
            valid_loss.append("softmax_loss")
        if "dice_loss" in loss_type:
            avg_loss_list.append(multi_dice_loss(logits, label, mask))
            loss_valid = True
            valid_loss.append("dice_loss")
        if "bce_loss" in loss_type:
            avg_loss_list.append(multi_bce_loss(logits, label, mask))
            loss_valid = True
            valid_loss.append("bce_loss")
        if not loss_valid:
            raise Exception(
                "SOLVER.LOSS: {} is set wrong. it should "
                "include one of (softmax_loss, bce_loss, dice_loss) at least"
                " example: ['softmax_loss'], ['dice_loss'], ['bce_loss', 'dice_loss']"
                .format(cfg.SOLVER.LOSS))

        invalid_loss = [x for x in loss_type if x not in valid_loss]
        if len(invalid_loss) > 0:
            print(
                "Warning: the loss {} you set is invalid. it will not be included in loss computed."
                .format(invalid_loss))

        avg_loss = 0
        for i in range(0, len(avg_loss_list)):
            avg_loss += avg_loss_list[i]

    #get pred result in original size
    if isinstance(logits, tuple):
        logit = logits[0]
    else:
        logit = logits

    if logit.shape[2:] != label.shape[2:]:
        logit = fluid.layers.resize_bilinear(logit, label.shape[2:])

    # return image input and logit output for inference graph prune
    if ModelPhase.is_predict(phase):
        # ????????????????????????dice_loss???bce_loss?????????logit??????????????????????????????????????????
        if class_num == 1:
            logit = sigmoid_to_softmax(logit)
        else:
            logit = softmax(logit)

        # ??????????????????
        logit = fluid.layers.slice(
            logit, axes=[2, 3], starts=[0, 0], ends=valid_shape)

        logit = fluid.layers.resize_bilinear(
            logit, out_shape=origin_shape, align_corners=False, align_mode=0)
        logit = fluid.layers.argmax(logit, axis=1)
        return origin_image, logit

    if class_num == 1:
        out = sigmoid_to_softmax(logit)
        out = fluid.layers.transpose(out, [0, 2, 3, 1])
    else:
        out = fluid.layers.transpose(logit, [0, 2, 3, 1])

    pred = fluid.layers.argmax(out, axis=3)
    pred = fluid.layers.unsqueeze(pred, axes=[3])
    if ModelPhase.is_visual(phase):
        if class_num == 1:
            logit = sigmoid_to_softmax(logit)
        else:
            logit = softmax(logit)
        return pred, logit

    if ModelPhase.is_eval(phase):
        return data_loader, avg_loss, pred, label, mask

    if ModelPhase.is_train(phase):
        decayed_lr = None
        if not cfg.SLIM.KNOWLEDGE_DISTILL:
            optimizer = solver.Solver(main_prog, start_prog)
            decayed_lr = optimizer.optimise(avg_loss)
        # optimizer = solver.Solver(main_prog, start_prog)
        # decayed_lr = optimizer.optimise(avg_loss)
        return data_loader, avg_loss, decayed_lr, pred, label, mask, image


def to_int(string, dest="I"):
    return struct.unpack(dest, string)[0]


def parse_shape_from_file(filename):
    with open(filename, "rb") as file:
        version = file.read(4)
        lod_level = to_int(file.read(8), dest="Q")
        for i in range(lod_level):
            _size = to_int(file.read(8), dest="Q")
            _ = file.read(_size)
        version = file.read(4)
        tensor_desc_size = to_int(file.read(4))
        tensor_desc = VarType.TensorDesc()
        tensor_desc.ParseFromString(file.read(tensor_desc_size))
    return tuple(tensor_desc.dims)
