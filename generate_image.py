# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib
import time
import datetime
import glob
import random
import cv2

from PIL import Image
from IPython import display

def saved_images(model, test_input, number):
    prediction = model(test_input, training=False)
    if CHANNEL == 1:
        predict_image = prediction[0].numpy().flatten().reshape(256, 256) # モノクロ画像の場合
        plt.imsave('predict_images/predict_{}.jpg'.format(number), predict_image, cmap="gray")
    else:
        predict_image = prediction[0].numpy().flatten().reshape(256, 256, 3)
        plt.imsave('predict_images/predict_{}.jpg'.format(number), predict_image)


# 学習に使うデータローダー
def read_single_image(dataset, batch_size=1, size=256):
        train_image_dataset = dataset

        batch_input = [] # 入力画像のバッチ

        for pd in train_image_dataset:
            input = pd[0] # 実態は入力画像のパス
            
            # モノクロ画像の場合、モノクロ画像として画像を取得
            if CHANNEL == 1:
                input = Image.open(input).convert("L")
                input = input.resize((256, 256))
            else:
                input = Image.open(input)
                input = input.resize((256, 256))
            # 入力画像の形式（シェイプ）をtensorflow、kerasで学習,推論するために変換
            input = np.reshape(input, [1, size, size, CHANNEL])
            # 入力画像をテンソルに変換
            input = tf.cast(tf.convert_to_tensor(np.asarray(input)), dtype=tf.float32) / 255.

            # 入力画像のバッチ
            batch_input += [input]
            
            # 今回はバッチサイズは1なので、単純にペアの画像を返しているだけ
            if len(batch_input) ==  batch_size:
                batch_input = tf.concat(batch_input, axis=0)

                yield {'input': batch_input}
                batch_input = []

def test_data_loader(batch_size=1):
    input_paths = sorted(glob.glob(os.path.join(str(PATH), '*.png'))) # QRコード画
    # input_paths = sorted(glob.glob(os.path.join(str(PATH) + "/test/" , 'tr*.jpg'))) # 数字画像
    records = []
    for input in input_paths:
        records += [[input]]

    return read_single_image(records, batch_size=batch_size)

# デコーダーの定義に使用
def upsample(filters, size,  dropout=0.5, max_pool=True, batch_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        # 畳み込み層の追加（アップサンプルに使用）
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)
    )
    
    # max poolingを行う場合
    if max_pool:
        result.add(tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=None, padding='same'))

    # バッチノルムを行う場合
    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())

    # ドロップアウトを行う場合
    if dropout != None:
        result.add(tf.keras.layers.Dropout(dropout))
    result.add(tf.keras.layers.ReLU())

    return result

# エンコーダーの定義に使用
def downsample(filters, kernel_size, strides=2, dropout=0.5, max_pool=True, batch_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        # 畳み込み層の追加
        tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                                kernel_initializer=initializer, use_bias=False))
    # max poolingを行う場合
    if max_pool:
        result.add(tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=None, padding='same'))

    # バッチノルムを行う場合
    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())
    
    # ドロップアウトを行う場合
    if dropout != None:
        result.add(tf.keras.layers.Dropout(dropout))

    result.add(tf.keras.layers.LeakyReLU())
    return result

# ジェネレーターのネットワークの定義
def Generator(image_shape):
    initializer = tf.random_normal_initializer(0., 0.02)
    # 入力画像
    input_image = keras.layers.Input(shape=image_shape, name='input_image')
    x = input_image

    # エンコーダーの定義
    enc1 = downsample(n_E1, kernel_size_E1, stride_E1, DropOut_E1, MaxPooling_E1, BatchNorm_E1)(x) # 正体は単純な畳み込み層
    enc2 = downsample(n_E2, kernel_size_E2 ,stride_E2, DropOut_E2, MaxPooling_E2, BatchNorm_E2)(enc1)
    enc3 = downsample(n_E3, kernel_size_E3, stride_E3, DropOut_E3, MaxPooling_E3, BatchNorm_E3)(enc2)
    enc4 = downsample(n_E4, kernel_size_E4, stride_E4, DropOut_E4, MaxPooling_E4, BatchNorm_E4)(enc3)
    enc5 = downsample(n_E5, kernel_size_E5 ,stride_E5, DropOut_E5, MaxPooling_E5, BatchNorm_E5)(enc4)
    enc6 = downsample(n_E6, kernel_size_E6 ,stride_E6, DropOut_E6, MaxPooling_E6, BatchNorm_E6)(enc5)
    enc7 = downsample(n_E7, kernel_size_E7 ,stride_E7, DropOut_E7, MaxPooling_E7, BatchNorm_E7)(enc6)
    enc8 = downsample(n_E8, kernel_size_E8, stride_E8, DropOut_E8, MaxPooling_E8, BatchNorm_E8)(enc7)

    # デコーダーの定義
    dec1 = upsample(n_E7, kernel_size_E7, DropOut_E7, MaxPooling_E7, BatchNorm_E7) # 正体は単純な畳み込み層
    dec2 = upsample(n_E6, kernel_size_E6, DropOut_E6, MaxPooling_E6, BatchNorm_E6)
    dec3 = upsample(n_E5, kernel_size_E5, DropOut_E5, MaxPooling_E5, BatchNorm_E5)
    dec4 = upsample(n_E4, kernel_size_E4, DropOut_E4, MaxPooling_E4, BatchNorm_E4)
    dec5 = upsample(n_E3, kernel_size_E3, DropOut_E3, MaxPooling_E3, BatchNorm_E3)
    dec6 = upsample(n_E2, kernel_size_E2, DropOut_E2, MaxPooling_E2, BatchNorm_E2)
    dec7 = upsample(n_E1, kernel_size_E1, DropOut_E1, MaxPooling_E1, BatchNorm_E1)
    

    # 画像を拡大する場合コメントアウト
    # zoom = upsample(CHANNEL, 4)

    # バイパスするエンコーダーの特徴量
    enc_value_list = [enc7, enc6, enc5, enc4, enc3, enc2, enc1]

    # バイパスするデコーダーの特徴量
    dec_value_list = [dec1, dec2, dec3, dec4, dec5, dec6, dec7]

    # バイパスを行うか判定するフラグ
    bipass_list = [Bipass_7, Bipass_6, Bipass_5, Bipass_4, Bipass_3, Bipass_2, Bipass_1]

    # バイパス処理のためにエンコーダーの最終出力をxとする
    x = enc8    

    # バイパス処理を行っている部分
    for dec, enc, bipass in zip(dec_value_list, enc_value_list, bipass_list):
        x = dec(x)
        # バイパスを行うかを判定する
        if bipass:
            x = tf.keras.layers.Concatenate()([x, enc]) # バイパスしている行

    # 画像を拡大する場合コメントを外す
    # if expantion:
    #     x = zoom(x)

    # 出力のチャネル数、カラーだと3、モノクロだと1
    OUTPUT_CHANNELS = CHANNEL

    # 層の乱数を初期化するもの
    initializer = tf.random_normal_initializer(0., 0.02)

    # 最終出力層
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')
    x = last(x)

    return tf.keras.Model(inputs=input_image, outputs=x)

CHANNEL = 1
# 入力画像サイズ
G_input_dim = (256, 256, CHANNEL) 
EPOCH = 20

# 第１層目のパラメータ
n_E1 = 32 # チャネル数
m_E1 = 128 # 画素数
stride_E1 = 2 # ストライドのサイズ
kernel_size_E1 = 4 # カーネルサイズ
MaxPooling_E1 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E1 = "Leaky_ReLu" # 活性化関数
BatchNorm_E1 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E1 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_1 = True # バイパスの設定（バイパスを行わない場合はNoneにする） 

# 第２層目のパラメーター
n_E2 = 64 # チャネル数
m_E2 = 64 # 画素数
stride_E2 = 2 # ストライドのサイズ
kernel_size_E2 = 4 # カーネルサイズ
MaxPooling_E2 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E2 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E2 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E2 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_2 = True # バイパスの設定（バイパスを行わない場合はNoneにする） 

# 第3層目のパラメーター
n_E3 = 128 # チャネル数
m_E3 = 128 # 画素数
stride_E3 = 2 # ストライドのサイズ
kernel_size_E3 = 4 # カーネルサイズ
MaxPooling_E3 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E3 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E3 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E3 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_3 = True # バイパスの設定（バイパスを行わない場合はNoneにする） 

# 第4層目のパラメーター
n_E4 = 256 # チャネル数
m_E4 = 256 # 画素数
stride_E4 = 2 # ストライドのサイズ
kernel_size_E4 = 4 # カーネルサイズ
MaxPooling_E4 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E4 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E4 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E4 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_4 = True # バイパスの設定（バイパスを行わない場合はNoneにする） 

# 第5層目のパラメーター
n_E5 = 512 # チャネル数
m_E5 = 512 # 画素数
stride_E5 = 2 # ストライドのサイズ
kernel_size_E5 = 4 # カーネルサイズ
MaxPooling_E5 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E5 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E5 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E5 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_5 = True # バイパスの設定（バイパスを行わない場合はNoneにする）

# 第6層目のパラメーター
n_E6 = 512 # チャネル数
m_E6 = 512 # 画素数
stride_E6 = 2 # ストライドのサイズ
kernel_size_E6 = 4 # カーネルサイズ
MaxPooling_E6 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E6 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E6 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E6 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_6 = True # バイパスの設定（バイパスを行わない場合はNoneにする）

# 第7層目のパラメーター
n_E7 = 512 # チャネル数
m_E7 = 512 # 画素数
stride_E7 = 2 # ストライドのサイズ
kernel_size_E7 = 4 # カーネルサイズ
MaxPooling_E7 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E7 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E7 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E7 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）
Bipass_7 = True # バイパスの設定（バイパスを行わない場合はNoneにする）


# 第8層目のパラメーター
n_E8 = 512 # チャネル数
m_E8 = 512 # 画素数
stride_E8 = 2 # ストライドのサイズ
kernel_size_E8 = 4 # カーネルサイズ
MaxPooling_E8 = True # MaxPoolingの設定（MaxPoolingを行わない場合はNoneにする）🇲
ActivationFunc_E8 = "Leaky_ReLu" # 活性化関数
alfa = 0.2
BatchNorm_E8 = True # 　バッチ正規化の設定（正規化を行わない場合はNoneにする）
DropOut_E8 = 0.5 # ドロップアウトの設定（ドロップアウトを行わない場合はNoneにする）

# 画像拡大パラメータ
expantion = True # 画像をconv2dで２倍に拡大する

PATH = "bottle/test/broken_large"
generator = Generator(G_input_dim)
generator.load_weights('checkpoints/cp-20.h5')
test_generator = test_data_loader(batch_size=1)
for i, test_loader_dict in enumerate(test_generator):
    example_input = test_loader_dict['input'] # 入力画像を取得する
    saved_images(generator, example_input, i)