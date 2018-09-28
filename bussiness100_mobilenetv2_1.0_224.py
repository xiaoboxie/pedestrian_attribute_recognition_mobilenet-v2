#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 09:47:36 2018

@author: bobby
"""

"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

#from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import graph_util
#import VGG

#import sys
#sys.path.append("/home/fs/work/models/research/slim")#module and script not in the same path

#import tensorflow as tf
from nets.mobilenet import mobilenet_v2
#from nets import alexnet

import tensorflow.contrib.slim as slim

import re
"""
Configuration Part.
"""
tf.reset_default_graph()
momentum_path = re.compile(r'(MobilenetV2/Logits(?!/))|(MobilenetV2/(.+?/){2,6}?Momentum)')
excludeD = ["MobilenetV2/Logits", "MobilenetV2/Conv/BatchNorm/gamma/Momentum",
            "MobilenetV2/Conv/BatchNorm/beta/Momentum",
            "MobilenetV2/Conv_1/BatchNorm/beta/Momentum", "MobilenetV2/Conv/weights/Momentum",
            "MobilenetV2/Conv/BatchNorm/beta/Momentum", "MobilenetV2/Conv/BatchNorm/gamma/Momentum",
            "MobilenetV2/Conv_1/BatchNorm/gamma/Momentum", "MobilenetV2/expanded_conv_9/project/weights/Momentum",
            "MobilenetV2/Conv/weights/Momentum", "MobilenetV2/Conv_1/BatchNorm/beta/Momentum",
            "MobilenetV2/Conv_1/weights/Momentum", "MobilenetV2/Logits/Conv2d_1c_1x1/biases/Momentum",
            "MobilenetV2/Logits/Conv2d_1c_1x1/weights/Momentum",
            "MobilenetV2/expanded_conv/depthwise/BatchNorm/beta/Momentum",
            'MobilenetV2/Conv/BatchNorm/gamma/Momentum', 'MobilenetV2/Conv/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_9/project/weights/Momentum', 'MobilenetV2/Conv/weights/Momentum',
            'MobilenetV2/Conv_1/BatchNorm/gamma/Momentum', 'MobilenetV2/Conv_1/BatchNorm/beta/Momentum',
            'MobilenetV2/Logits/Conv2d_1c_1x1/biases/Momentum', 'MobilenetV2/Conv_1/weights/Momentum',
            'MobilenetV2/expanded_conv/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/Logits/Conv2d_1c_1x1/weights/Momentum',
            'MobilenetV2/expanded_conv/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_9/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv/project/weights/Momentum',
            'MobilenetV2/expanded_conv_9/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_1/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_1/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_9/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_1/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_1/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_1/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_1/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_1/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_10/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_1/project/weights/Momentum',
            'MobilenetV2/expanded_conv_9/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_10/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_10/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_10/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_10/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_10/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_10/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_10/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_10/project/weights/Momentum',
            'MobilenetV2/expanded_conv_9/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_11/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_11/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_11/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_11/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_11/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_11/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_11/project/weights/Momentum',
            'MobilenetV2/expanded_conv_12/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_12/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_12/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_12/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_12/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_11/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_11/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_9/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_12/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_12/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_12/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_12/project/weights/Momentum',
            'MobilenetV2/expanded_conv_13/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_13/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_13/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_13/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_13/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_13/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_13/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_9/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_13/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_13/project/weights/Momentum',
            'MobilenetV2/expanded_conv_14/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_14/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_14/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_9/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_14/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_14/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_14/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_14/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_14/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_15/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_15/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_8/project/weights/Momentum',
            'MobilenetV2/expanded_conv_14/project/weights/Momentum',
            'MobilenetV2/expanded_conv_15/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_15/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_15/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_15/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_15/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_15/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_15/project/weights/Momentum',
            'MobilenetV2/expanded_conv_16/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_16/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_16/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_16/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_16/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_16/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_8/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_16/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_16/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_16/project/weights/Momentum',
            'MobilenetV2/expanded_conv_2/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_2/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_2/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_2/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_2/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_8/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_2/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_2/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_2/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_2/project/weights/Momentum',
            'MobilenetV2/expanded_conv_3/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_3/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_8/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_3/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_3/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_3/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_3/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_3/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_3/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_4/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_4/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_3/project/weights/Momentum',
            'MobilenetV2/expanded_conv_4/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_8/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_4/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_4/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_4/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_4/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_4/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_8/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_4/project/weights/Momentum',
            'MobilenetV2/expanded_conv_5/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_5/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_8/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_5/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_5/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_5/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_5/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_5/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_5/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_5/project/weights/Momentum',
            'MobilenetV2/expanded_conv_6/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_6/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_6/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_6/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_6/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_6/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_6/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_8/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_6/project/weights/Momentum',
            'MobilenetV2/expanded_conv_6/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_7/depthwise/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_7/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_7/depthwise/depthwise_weights/Momentum',
            'MobilenetV2/expanded_conv_7/expand/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_7/expand/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_8/depthwise/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_7/expand/weights/Momentum',
            'MobilenetV2/expanded_conv_7/project/BatchNorm/beta/Momentum',
            'MobilenetV2/expanded_conv_7/project/BatchNorm/gamma/Momentum',
            'MobilenetV2/expanded_conv_7/project/weights/Momentum',
            'MobilenetV2/Conv_1/weights',
            'MobilenetV2/Conv_1/BatchNorm/moving_variance',
            'MobilenetV2/Conv_1/BatchNorm/gamma',
            'MobilenetV2/Conv_1/BatchNorm/moving_mean',
            'MobilenetV2/Conv_1/BatchNorm/beta'
            ]

weight = np.array([0.04824719596877834, 0.06367943523677123, 0.010778405370293202, 0.018100577439585954, 0.018685370713412777, 0.010900966834867687, 0.02179843191360467, 0.8078096165226861, 0.06202310458695036, 0.013457248810278355,
 0.05305510713622882, 0.011737886550104878, 0.011769402355281174, 0.060723953062460825, 0.12349643346138088, 0.019749904577145438, 0.015299172535026316, 0.036523316443196265, 0.014836940725773975, 0.014224133402901556, 0.012749894071877047, 0.024554313988465214,
 0.014395719453305833, 0.5114034688396231, 0.23887929796793092, 0.018167110806069243, 0.02615111478406421, 0.012455746556898285, 0.043512821680072554, 0.011006019518788672, 0.5026490785128742, 0.13439740029624858,
 0.012781409877053342, 0.19487973218569113, 0.1526065321758862, 0.06111264799296847, 0.029414751497876184, 0.04883549099873587, 0.5131508451488421, 0.39436777543938284, 0.515332439218268, 0.014861453018688872,
 0.07543833232366032, 0.03998305150032742, 0.026637858886231446, 0.01279541690157614, 0.01942073950085968, 0.9011629332110053, 0.24950712782460405, 0.09264596194991788,
 0.40386453806584, 0.022635351628841865, 0.057635404155183825, 0.05503359935007406, 0.03537123867619611, 0.038519317437695, 0.04478746091164719, 0.019277167499501, 0.02435121213288464, 0.0547394518350953, 0.020072066141169798,
 0.01913359549814232, 0.09220824243358045, 0.06825623049959555, 0.03949630739816018, 0.02460684033042571, 0.06385452304330622, 0.0346743892061869, 0.021871968792349362, 0.035126115747047146, 0.09227127404393304, 0.02990149560004342, 0.360159119798579,
 0.026861971278596217, 0.013821431447871107, 0.023003036022565315, 0.011429732010603318, 0.013576308518722139, 0.05581098921108936, 0.016524787180771158, 0.014525284430141717, 0.01273238529122355, 0.016370709911020378, 0.012998518757156715, 0.016258653714837992, 0.02058332253625193,
 0.14092117196774182, 0.014325684330691842, 0.5902560133907154, 0.022558312993966476, 0.8097776034681393, 0.034114108225274976, 0.018643349639844382, 0.03192901239971846, 0.05867892748213229, 0.02429868579092415, 0.08861193888735201,
 0.013464252322539753, 0.49596072430323807, 0.0146443441385855, 0.3873187403482847, 0.03378144139285852, 0.030783938144979706, 0.04030171130822107, 0.03496853672116566,
 0.014994519751655454, 0.016661355669868438, 0.8285084970112512, 0.03988150057253713, 0.6329844416975113, 0.3271340577299516, 0.0439295306596258, 0.726810495463475, 0.026256167467985196,
 0.016853952257056912, 0.017764408851038797, 0.1434004153082771, 0.02498502999254126, 0.22395481333888945, 0.05326521250407079, 0.0648525235405556, 0.08877652142549489,
 0.018310682807427925, 0.06173946234036369, 0.12701219661660323, 0.31354724394283734, 0.011156595032408753, 0.01226314996970981, 0.013947494668576291, 0.01117410381306225])#all attribute

#weight = np.array([0.9452927107406324,0.8879514789435973,0.8872349658394809,0.8705936967030306,0.7787387351020779,0.6955122059521046,0.6266159388844598, 0.5837260699760826,
# 0.5624829701991099, 0.5118628331533641, 0.5057876093691657, 0.5023463281226348, 0.46910415678517725,0.41970511953658757, 0.41585007720176403,
# 0.4025693554409583,0.3391428081258641,0.3049822890070743,0.2897538626111352,0.26963094529271076,0.24767133241162165,0.22266401590457258,
# 0.16097324681353503,0.14966041315558426,0.14794481839924917,0.1433934464280308,0.14044666014067878,0.13538060974255986,0.10409623477409653])#splitH

#weight = np.array([0.11207729468599034, 0.10598917408765497, 0.10183342063907805, 0.10066934404283802, 0.08165997322623829, 0.08159012863046389,
#                   0.07993713986380303, 0.07822594726733019, 0.0751993481171061, 0.07342995169082125, 0.07298760258425005, 0.07244048658401722, 
#                   0.07223095279669402, 0.0674465979861475, 0.06624759909202026, 0.06565392002793784, 0.06390780513357779, 0.056364588789942376,
#                   0.05409463942727431, 0.04453757057214365, 0.044118502997497235, 0.041580816017693964, 0.04138292299633316, 0.04096385542168675,
#                   0.0394971189104243, 0.03789069320761306, 0.03768115942028986, 0.03588848146208021, 0.0356673069087946, 0.034561434142366565,
#                   0.03453815261044177, 0.031860776439089696, 0.030231069204353648, 0.026948373202956753, 0.024480530818927884, 0.023479424946161457,
#                   0.023386298818462256, 0.022245503754147022, 0.0213608055410046, 0.021139630987718993, 0.019614690646644548, 0.019160700774110938,
#                   0.018636866305802923, 0.01820615796519411, 0.0179384203480589, 0.01750771200745009, 0.017111925964728478, 0.016331994645247656,
#                   0.01632035387928526, 0.015261044176706828, 0.015051510389383621, 0.014807054304173214, 0.013747744601594785, 0.01353821081427158,
#                   0.012397415749956347, 0.011850299749723531])#splitM

#weight = np.array([0.06541473217379253, 0.05897027233039984, 0.05619846164506964, 0.05384242256253898, 0.05141708821287506, 0.04725937218487977,
#                   0.04441826623241633, 0.04039914073868755, 0.037904511121890375, 0.037558034786224104, 0.037211558450557826, 0.036241424710692256,
#                   0.034301157230961124, 0.03340031875822881, 0.0324994802854965, 0.031182870209964658, 0.030974984408564894, 0.030351327004365602,
#                   0.028549650058900977, 0.027233039983369137, 0.026401496777770078, 0.026055020442103803, 0.025500658305037765, 0.023421800291040122,
#                   0.022728847619707573, 0.02106576120850946, 0.02064998960570993, 0.019679855865844365, 0.01773958838611323, 0.017462407317580208,
#                   0.015244958769316056, 0.01372046289238445, 0.013166100755318412, 0.012819624419652138, 0.012819624419652138, 0.010324994802854965,
#                   0.010047813734321946, 0.009354861062989397, 0.008038250987457557, 0.007483888850391519, 0.005474326103527129, 0.004088420760862033,
#                   0.00332617282239623, 0.002148153281130899, 0.0013166100755318412])

#weight = np.array([0.37343123025697944, 0.1983693332194997, 0.13079484333646377, 0.12665414496713054, 0.0988218219072825, 0.07192862631264407])#test
#weight = np.array([0.04637868943046055, 0.06754825555915474, 0.0005430536765686629, 0.01377641432137345, 0.015634229530687296,
#                   0.0008860349459804501, 0.017758807949543643, 0.8374745145862312, 0.07181646691183476, 0.003391703664183228,
#                   0.06283226310474267, 0.0016291610297059888, 0.0012575979878432194, 0.07559878813284808, 0.13398182199272118, 
#                   0.011318381890588975, 0.005859263352451363, 0.029620243516701283, 0.005287627903431718, 0.0038109029934643014,
#                   0.0032964310893466207, 0.020407385530001335, 0.004534974562222518, 0.5653569863379128, 0.2618280901659648, 
#                   0.009022312837026734, 0.020778948571864103, 0.005430536765686629, 0.040366989958270615, 0.0016863245746079533,
#                   0.509184276214249, 0.14438558716487873, 0.007316933747451459, 0.21426802080753035, 0.15870505516282082,
#                   0.03567957927630952, 0.0222842552542825, 0.033431146510165585, 0.5356319429888912, 0.40903374554600713, 
#                   0.5032106857719937, 0.010518092261961471, 0.07723747642003773, 0.031030277624283075, 0.016272555782092565,
#                   0.0042586840951963564, 0.012032926201863532, 0.9364055562965645, 0.27793868257083515, 0.1023227453745165,
#                   0.43248032621329624, 0.01048951048951049, 0.032764238486309334, 0.05193308054343477, 0.02615232179264877,
#                   0.0353461252643814, 0.03057296926506736, 0.03375507326461005, 0.017539681027419447, 0.06494731426611536,
#                   0.008679331567614947, 0.019492768811569902, 0.09546311998628075, 0.06905356224157314, 0.03655608696480631, 
#                   0.01443379508774604, 0.06738629218193251, 0.03753739448562337, 0.016072483374935692, 0.03446009031840094, 
#                   0.08835578590346983, 0.022322364284217144, 0.37394485623368456, 0.0356605247613422, 0.003124940454640727, 
#                   0.01479583087212515, 0.0006097444789542882, 0.002619995808006707, 0.06373735256569044, 0.008622168022712982,
#                   0.003601303328823765, 0.0021055239038890263, 0.006526171376307616, 0.004496865532287875, 0.007002534250490654,
#                   0.0088603494598045, 0.15808578342638288, 0.003829957508431623, 0.6763209542501095, 0.019016405937386863, 
#                   0.8328347401916885, 0.019168842057125435, 0.005659190945294487, 0.024446942703073495, 0.06251786360778186,
#                   0.03635601455764943, 0.09414835845353557, 0.010889655303824242, 0.48992968883977056, 0.011232636573236029, 
#                   0.3937996608296336, 0.04151978811379357, 0.014700558297288542, 0.03047769669023075, 0.021217202416112496,
#                   0.01251881633353023, 0.012861797602942017, 0.8667041405461025, 0.037889903012518815, 0.6280463405804005,
#                   0.33406375640708064, 0.052380861645166824, 0.74286884777348, 0.016663173338922657, 0.003858539280882605,
#                   0.015491320668432385, 0.1488538709247156, 0.019883386368399994, 0.24608906080295725, 0.05607743754882719,
#                   0.06864389016977573, 0.09679693603399325, 0.014567176692517291, 0.06258455441016748, 0.13531563804043367,
#                   0.31074103008707915, 0.000847925916045807, 0.002343705340980545, 0.00481126502924868, 0.0011813799279739334])
#weight = np.array([0.04639504407910412, 0.06757207529187514, 0.0005432451751250894, 0.013781272337383845, 0.0156397426733381, 
#                   0.0008863473909935668, 0.01776507028830117, 0.837417202763879, 0.07157493447700738, 0.003392899690254944, 
#                   0.06285441982368359, 0.001629735525375268, 0.0012580414581844174, 0.0756254467476769, 0.13368596616630926,
#                   0.011322373123659756, 0.00586132952108649, 0.029630688587086014, 0.0052894924946390275, 0.003812246842983083, 
#                   0.003297593519180367, 0.020414581844174412, 0.004536573743149869, 0.5658136764355493, 0.26167262330235885,
#                   0.00902549440076245, 0.02078627591136526, 0.0054324517512508936, 0.040381224684298306, 0.0016869192280200144,
#                   0.5092685251370026, 0.1444269716464141, 0.007319513938527519, 0.2140385989992852, 0.1590945913747915, 
#                   0.03569216106742912, 0.02229211341434358, 0.0334429354300691, 0.5354395997140815, 0.4094829640219204,
#                   0.5027305218012866, 0.01052180128663331, 0.07726471289015964, 0.031041219918989754, 0.016278294019537766,
#                   0.004260185847033596, 0.012037169406719086, 0.9363831308077198, 0.27728377412437455, 0.10235882773409578, 
#                   0.4330331188944484, 0.010493209435310936, 0.032775792232547056, 0.051951393852751965, 0.026161543959971407, 
#                   0.03535858946866809, 0.030583750297831783, 0.03376697641172266, 0.01754586609482964, 0.0649702168215392,
#                   0.008682392184893972, 0.01949964260185847, 0.09499166071003097, 0.06907791279485347, 0.03656897784131523,
#                   0.014438884917798427, 0.06741005480104836, 0.03755063140338337, 0.016078151060281154, 0.03447224207767453,
#                   0.08819633071241363, 0.02233023588277341, 0.3744198236835835, 0.0356730998332142, 0.003126042411246128, 
#                   0.014801048367881821, 0.0006099594948772933, 0.0026209197045508697, 0.06375982844889207, 0.008625208482249225,
#                   0.0036025732666190137, 0.0021062663807481534, 0.00652847271860853, 0.004498451274720038, 0.0070050035739814154, 
#                   0.008863473909935669, 0.15752203955206098, 0.0038313080771979987, 0.6768263045032166, 0.019023111746485584, 
#                   0.832775792232547, 0.01917560162020491, 0.005661186561829878, 0.02445556349773648, 0.06253990945913748,
#                   0.036368834882058616, 0.09418155825589707, 0.01089349535382416, 0.4899785561115082, 0.011236597569692638, 
#                   0.3937097927090779, 0.04153442935430069, 0.014705742196807243, 0.03048844412675721, 0.021224684298308316,
#                   0.012523230879199428, 0.012866333095067906, 0.8666571360495592, 0.037903264236359306, 0.6278484631879914, 
#                   0.33424827257564926, 0.052399332856802476, 0.7425780319275673, 0.01666904932094353, 0.0038598999285203717, 
#                   0.015496783416726233, 0.14910650464617584, 0.019890397903264235, 0.2469192280200143, 0.056649988086728616, 
#                   0.06813438170121515, 0.09700262091970455, 0.014572313557302836, 0.06256850131045985, 0.13513461996664283,
#                   0.3099547295687396, 0.0008672861567786515, 0.0023350011913271383, 0.004708124851084108, 0.0011532046700023827])
#weight = np.array([0.04251548057076191, 0.07076191330880373, 0.00021313829309880643, 0.005406981961769721, 0.006136139280265638,
#                   0.0003477519518980526, 0.0184645068652966, 0.8561540877681055, 0.07023467647850669, 0.0033429058601812797,
#                   0.06384052768554249, 0.0016265817104908912, 0.0012115229291932155, 0.07703266624786861, 0.13840527685542492,
#                   0.01141972538813605, 0.0058669119626671455, 0.029132639325136857, 0.005552813425468904, 0.003791618056178767, 
#                   0.003410212689580903, 0.02037153369828592, 0.004621735618774118, 0.5601386520685632, 0.26748855783900205,
#                   0.00954635196984654, 0.021437225163779952, 0.0021313829309880642, 0.03987929641927668, 0.0006618504890962936, 
#                   0.5095575697747465, 0.14642600735888, 0.0028717580543839182, 0.21805169164497892, 0.15099165395315445,
#                   0.036312034461096654, 0.022536570043973796, 0.020584671991384727, 0.5515233779054115, 0.41637126447096834,
#                   0.5008076819527955, 0.004128152203176882, 0.07869290137305932, 0.03070313201112806, 0.016490173202907656,
#                   0.0016714529300906398, 0.011946962218433096, 0.9391882796374406, 0.28381046396841064, 0.09813335726465046, 
#                   0.4481400879475904, 0.005261150498070538, 0.033283227138113616, 0.05212913937000808, 0.025969218343354573,
#                   0.03437135421340752, 0.018902001256394147, 0.01324822758682581, 0.017959705644799425, 0.0696064794041102, 
#                   0.009097639773849054, 0.007650542941757157, 0.09802117921565108, 0.07246701965359419, 0.039475455442878934,
#                   0.014706542223817643, 0.06980839989230907, 0.04006999910257561, 0.017286637350803196, 0.03651395494929552,
#                   0.08818316431840617, 0.02359104370456789, 0.382314008794759, 0.034584492506506324, 0.0028268868347841695, 
#                   0.014504621735618775, 0.0005384546351969847, 0.0020752939064883785, 0.06158574890065512, 0.00871623440725119,
#                   0.003185856591582159, 0.0020752939064883785, 0.006023961231266266, 0.00421789464237638, 0.007190612940859733, 
#                   0.008323611235753387, 0.1590572556762093, 0.003679440007179395, 0.6814143408417841, 0.01754464686350175, 
#                   0.8746522480481019, 0.010589607825540697, 0.0022211253701875616, 0.015727362469711927, 0.06499596159023602,
#                   0.014269047832720093, 0.09701157677465673, 0.004273983666876066, 0.4880193843668671, 0.004408597325675312, 
#                   0.4062864578659248, 0.042919321547159654, 0.015738580274611863, 0.03330566274791349, 0.022626312483173292, 
#                   0.004913398546172485, 0.005048012204971731, 0.8754487121959975, 0.038062012025486855, 0.6414789553980077,
#                   0.32045903257650543, 0.054316611325495825, 0.7643700080768195, 0.006539980256663376, 0.0015144036614915194, 
#                   0.006080050255765952, 0.15413263932513685, 0.01304630709862694, 0.24676927218881808, 0.05725567620927937, 
#                   0.07101992282150228, 0.0966974782374585, 0.015244996859014629, 0.06391905231984206, 0.13855110831912412, 
#                   0.30177016961321007, 0.0007291573184959167, 0.0022211253701875616, 0.004700260253073679, 0.001121780489993718])
#weight = np.array([0.03824767133241162, 0.06365865719389248, 0.00019174294335509785, 0.004864215720903008, 
#                   0.0055201784218546586, 0.0003128437496846333, 0.01661099393486795, 0.8705936967030306, 0.06781645154453987, 
#                   0.0030073366905167976, 0.0574320574017822, 0.0014633014098152204, 0.0010899072569658193, 
#                   0.06929993642207667, 0.14044666014067878, 0.010273385070288927, 0.0052779768091955875, 0.026208232836483637,
#                   0.004995408261093339, 0.0034110060449485825, 0.0030678870936815656, 0.018326588691203037, 
#                   0.004157794350647385, 0.5837260699760826, 0.26963094529271076, 0.008588065515536224, 0.019285303407978526, 
#                   0.0019174294335509783, 0.03587611387512488, 0.0005954122977868827, 0.5118628331533641, 0.14966041315558426,
#                   0.0025834838683634234, 0.22266401590457258, 0.1433934464280308, 0.032666942507392195, 0.020274293326336398, 
#                   0.018518331634558134, 0.5624829701991099, 0.41970511953658757, 0.5057876093691657, 0.003713758060772421, 
#                   0.07079351303347428, 0.027621075576994884, 0.014834848775368096, 0.0015036683452583988, 
#                   0.010747696561746274, 0.9452927107406324, 0.01191833768959845, 0.01615686591113219, 0.06261920860623063,
#                   0.008184396161104439, 0.006882562493061933, 0.10409623477409653, 0.06519260074073327, 0.03551281145613628, 
#                   0.013230263091501751, 0.06280085981572495, 0.0360476733507584, 0.015551361879484514, 0.0328485937168865,
#                   0.09188523680253505, 0.02122291630925109, 0.41585007720176403, 0.031112815492829824, 0.002543116932920245,
#                   0.013048611882007448, 0.0004844032253181419, 0.0018669707642470052, 0.05540361889576248, 
#                   0.007841277209837422, 0.002866052416465673, 0.0018669707642470052, 0.005419261083246712, 
#                   0.0037944919316587784, 0.006468801404769353, 0.007488066524709611, 0.16097324681353503, 
#                   0.0033100887063406362, 0.6955122059521046, 0.01578347175828279, 0.8872349658394809, 0.009526596764590124, 
#                   0.0019981633044373354, 0.014148610872834061, 0.05847150598944405, 0.01283668547093076, 0.08727331442815191,
#                   0.0038449506009627515, 0.5023463281226348, 0.003966051407292287, 0.4025693554409583, 0.03861097375140023,
#                   0.014158702606694856, 0.029962357832699234, 0.020355027197222753, 0.004420179431028045, 
#                   0.004541280237357581, 0.8879514789435973, 0.03424125298967615, 0.6266159388844598, 0.3391428081258641, 
#                   0.048864175353967566, 0.7787387351020779, 0.005883480840843266, 0.0013623840712072741, 
#                   0.0054697197525506855, 0.14794481839924917, 0.011736686480104147, 0.24767133241162165, 0.05691737897488167,
#                   0.07073296263030951, 0.09716321361173064, 0.015177967726635114, 0.06327517130718228, 0.13538060974255986,
#                   0.3049822890070743, 0.0007972469750027752, 0.002220181449374817, 0.004511005035775196, 
#                   0.0011706411278521764])
weight = np.exp(weight)

print(weight)


# Path to the textfiles for the trainings and validation set
#train_after_augmentation=False
#if train_after_augmentation:
#    train_file = '/home/bobby/work/dataset/bussiness100/train_new.txt'   # trainlist_shuffle.txt
#else:
#    train_file = '/home/bobby/work/dataset/bussiness100/train_new (复件).txt'
train_file = '/home/bobby/work/dataset/bussiness100/train_undersample.txt'
val_file = '/home/bobby/work/dataset/bussiness100/val_new.txt'

# Learning params
learning_rate = 0.1
num_epochs = 200
batch_size = 64 
# Network params
num_classes = len(weight)
# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./bussiness100_mobilenetv2_1.0_224/tensorboard"
checkpoint_path = "./bussiness100_mobilenetv2_1.0_224/checkpoints_mobilenetv2_1.0_224"
pb_file_path = "./bussiness100_mobilenetv2_1.0_224/pb_file/bussiness100_mobilenetv2.pb"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)



# Place data loading and preprocessing on the cpu
#gpu_options = tf.GPUOptions(allow_growth=True)
#session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True,
                                 augmentation=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False,
                                  augmentation=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
#x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
x = tf.placeholder(tf.float32, [batch_size, 256, 96, 3], name='x')
y = tf.placeholder(tf.float32, [batch_size, num_classes], name='y')
keep_prob = tf.placeholder(tf.float32)


# # Note: arg_scope is optional for inference.
with slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
    score, endpoints = mobilenet_v2.mobilenet(x,num_classes,depth_multiplier=1.0,scope='MobilenetV2')

    # score = tf.sigmoid(score)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score,labels=y))
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=score, targets=y, pos_weight=weight))


# Train op

    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    # train_op = slim.train.AdamOptimizer(learning_rate).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.name_scope("train"):
    with tf.control_dependencies([tf.group(*update_ops)]):
        # train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)
        train_op = optimizer.minimize(loss)
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("test"):
    scoreTest = tf.sigmoid(score)
# Evaluation op: Accuracy of the model
def get_metrics(prediction, y_true):
    tp = tf.reduce_sum(tf.bitwise.bitwise_and(prediction, y_true))
    tp_fp = tf.reduce_sum(prediction)
    tp_fn = tf.reduce_sum(y_true)
    fn_fp = tf.reduce_sum(tf.bitwise.bitwise_xor(prediction, y_true))    
    return tp,tp_fp,tp_fn,fn_fp+tp
with tf.name_scope("accuracy"):
    att=[]
    score_acc = tf.sigmoid(score)
    score_acc = tf.round(score_acc)

    # correct_pred = tf.equal(score_accscore_acc, y)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    predict = tf.cast(score_acc, tf.int32)
    target = tf.cast(y, tf.int32)
    tmp = tf.reduce_sum(tf.bitwise.bitwise_and(predict, target))
    precision = tf.divide(tmp, tf.reduce_sum(predict))
    recall = tf.divide(tmp, tf.reduce_sum(target))

    #accuracy = accuracy + np.divide(tmp, np.sum(target+out-np.bitwise_and(target,out)))
    a = tf.add(predict,target)
    b = tf.bitwise.bitwise_and(target, predict)
    c = tf.subtract(a,b)

    accuracy = tf.divide(tmp, tf.reduce_sum(c))
    #attribute_metrics = {i: get_metrics(j,k) for i,j,k in zip(tf.range(tf.shape(target)[1]),tf.transpose(predict), tf.transpose(target))}
    
    for i in range(num_classes):
        att.append(get_metrics(tf.transpose(predict)[i],tf.transpose(target)[i]))
#with tf.name_scope("attribute_acc"):
#    score_acc_att = tf.sigmoid(score)
#    score_acc_att = tf.round(score_acc_att)
#    prediction = tf.cast(score_acc_att, tf.int32)
#    y_true = tf.cast(y, tf.int32)
#    attribute_metrics = get_metrics(precision, y_true)
    #attribute_metrics = {i: get_metrics(j,k) for i,j,k in zip(range(y_true.shape[1]),tf.transpose(prediction), tf.transpose(y_true))}
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('recall', recall)
tf.summary.scalar('precision', precision)


# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))



# Start Tensorflow session

with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # sess.run(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore)

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    variables_to_restore = slim.get_variables_to_restore(exclude=excludeD)
    # # print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
#    saver.restore(sess, "/home/bobby/work/code/bussiness100/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt")
    saver.restore(sess, "/home/bobby/work/code/bussiness100/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt")
    #variables_to_restore = slim.get_variables_to_restore()
    #saver = tf.train.Saver(variables_to_restore)
    # restore from checkpoint
    #saver.restore(sess, "/home/fs/work/finetune_alexnet_with_tensorflow/tmp/finetune_alexnet/checkpoints_mobilenetv2_1.0_224/model_epoch20.ckpt")

    saver = tf.train.Saver(max_to_keep=0)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(0,num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        print('#################################%s###################################'%(epoch+1))
        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            sess.run(train_op, feed_dict={x: img_batch,y: label_batch})
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,y: label_batch})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0.
        recall_val = 0.
        precision_val = 0.
        attr_val = np.zeros(shape=(num_classes,4))
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc,pre,rec,attribute_metrics = sess.run([accuracy,precision,recall,att], feed_dict={x: img_batch, y: label_batch})
#            attribute_metrics = sess.run(attribute_metrics)
#            with open('/home/bobby/work/code/bussiness100/attribute_metrics.txt', 'a+') as f:
#                f.write(str(attribute_metrics))
            # test = sess.run(scoreTest,feed_dict={x: img_batch,y: label_batch})
            # print("test {}".format(test))
            attribute_metrics = np.array(attribute_metrics)
            attr_val += attribute_metrics
            test_acc += acc
            recall_val += rec
            precision_val += pre
            test_count += 1
            # print("{} acc = {:.4f}".format(datetime.now(),acc))
        tp = attr_val[:,0]
        tp_fp = attr_val[:,1].clip(min=1e-8)#avoid divide zero
        tp_fn = attr_val[:,2].clip(min=1e-8)
        tp_fn_fp = attr_val[:,3]
        precision_n = tp/tp_fp
        recall_n = tp/tp_fn
        accuracy_n = tp/tp_fn_fp

        with open('/home/bobby/work/code/bussiness100/each_attribute_count.txt') as f:
            att_count = f.readlines()
            att_count = [i.strip() for i in att_count]
            
        with open('attribute_metrics/%s epoch_attribute_metrics_withrap' %(epoch+1), 'w') as f:
            for i in zip(precision_n,recall_n,att_count):#accuracy_n
                f.write(str(i)+'\n')
        test_acc /= test_count
        precision_val /= test_count
        recall_val /= test_count
        #print('each attribute metrics', attr_val)
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Validation recall = {:.4f}".format(datetime.now(),
                                                     recall_val))
        print("{} Validation precision = {:.4f}".format(datetime.now(),
                                                        precision_val))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
#        constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['output'])
#        with tf.gfile.GFile(pb_file_path, mode='wb') as f:
#            f.write(constant_graph.SerializeToString)


    
