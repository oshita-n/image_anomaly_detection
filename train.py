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

def load(image_path):
    # 入力画像のパス、正解画像のパス
    input_image_path, real_image_path = image_path
    
    # 入力画像の読み込み
    input_image = tf.io.read_file(input_image_path) # 入力画像の取得
    # JPEGの画像をuint8形式のテンソルに変換
    input_image = tf.io.decode_jpeg(input_image, channels=CHANNEL) 
    # 正解画像の読み込み
    real_image = tf.io.read_file(real_image_path)
    # JPEGの画像をuint8形式のテンソルに変換
    real_image = tf.io.decode_jpeg(real_image, channels=CHANNEL)

    # uint8のテンソルを学習しやすいようにfloat32のテンソルに変換
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

# 学習に使うデータローダー
def read_bd_rm(dataset, batch_size=1, size=256):
        train_image_dataset = dataset

        batch_input = [] # 入力画像のバッチ
        batch_real = [] # 正解画像のバッチ

        for pd in train_image_dataset:
            input = pd[0] # 実態は入力画像のパス
            real = pd[1] # 実態は正解画像のパス
            
            # モノクロ画像の場合、モノクロ画像として画像を取得
            if CHANNEL == 1:
                input = Image.open(input).convert("L")
                input = input.resize((256, 256))
                real = Image.open(real).convert("L")
                real = real.resize((256, 256))
            else:
                input = Image.open(input)
                input = input.resize((256, 256))
                real = Image.open(real)
                real = real.resize((256, 256))
            # 入力画像の形式（シェイプ）をtensorflow、kerasで学習するために変換
            input = np.reshape(input, [1, size, size, CHANNEL])
            # 正解画像の形式（シェイプ）をtensorflow、kerasで学習するために変換
            real = np.reshape(real, [1, size, size, CHANNEL])
            # 入力画像をテンソルに変換
            input = tf.cast(tf.convert_to_tensor(np.asarray(input)), dtype=tf.float32) / 255.
            # 正解画像をテンソルに変換
            real = tf.cast(tf.convert_to_tensor(np.asarray(real)), dtype=tf.float32) / 255.

            # 入力画像のバッチ
            batch_input += [input]
            # 正解画像のバッチ
            batch_real += [real]
            
            # 今回はバッチサイズは1なので、単純にペアの画像を返しているだけ
            if len(batch_input) ==  batch_size:
                batch_input = tf.concat(batch_input, axis=0)
                batch_real = tf.concat(batch_real, axis=0)

                yield {'input': batch_input, 'real': batch_real}
                batch_input = []
                batch_real = []

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
            # 入力画像の形式（シェイプ）をtensorflow、kerasで学習するために変換
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

def train_data_loader(batch_size=1):
    # 画像をファイル名でソートする
    input_paths = sorted(glob.glob(os.path.join(str(PATH), '*.png'))) # 入力画像のパスの配列
    real_paths = sorted(glob.glob(os.path.join(str(PATH), '*.png'))) # 正解画像のパスの配列
    
    # ペアの画像のパスを補完するリスト
    records = []
    
    # 画像のパスのペアを作成
    # 正解画像のフォルダに画像が一枚しかないため、QRコードの時だけこっちを使用
    for input in input_paths:
        records += [[input, real_paths[0]]]

    # QRコード以外（正解画像と入力画像の枚数が同じ場合）の場合は上をコメントアウトして、こっちを使用
    # for input, real in zip(input_paths, real_paths):
    #     records += [[input, real]]

    return read_bd_rm(records, batch_size=batch_size) # 画像のペアのパスをデータローダーに渡す

def test_data_loader(batch_size=1):
    input_paths = sorted(glob.glob(os.path.join(str(PATH) + "/test/" , '*.jpg'))) # QRコード画
    # input_paths = sorted(glob.glob(os.path.join(str(PATH) + "/test/" , 'tr*.jpg'))) # 数字画像
    records = []
    for input in input_paths:
        records += [[input]]

    return read_single_image(records, batch_size=batch_size)

# モデルの学習を行っている部分
def train_step(input_image, target, step, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # ジェネレーターに画像を入力し、生成画像を返してもらう
        gen_output = generator(input_image, training=True)

        # 入力画像と正解画像を入力し、ロス計算のための画像を返してもらう
        disc_real_output = discriminator([input_image, target], training=True)
        # 入力画像と生成画像を入力し、判別計算のための画像を返してもらう
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        # ジェネレーターのロス計算
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        # ディスクリミネーターのロス計算
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # ジェネレータの勾配
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    # ディスクリミネーターの勾配
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    # ジェネレータの勾配更新
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    # ディスクリミネーターの勾配更新
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    
    if step % 10 == 0:
        # ロスの出力
        print("epoch: {} step: {} gen_total_loss: {}".format(epoch, step, str(gen_total_loss.numpy())))
        print("epoch: {} step: {} gen_gan_loss: {}".format(epoch, step, str(gen_gan_loss.numpy())))
        print("epoch: {} step: {} disc_loss: {}".format(epoch, step, str(disc_loss.numpy())))

        # tensorboardでの学習経過を確認するための処理
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step)
            tf.summary.scalar('disc_loss', disc_loss, step)


def fit():
    for epoch in range(1, EPOCH+1):
        step = 1
        # 学習に使うジェネレーターの呼び出し（データセットのようなもの）
        train_generator = train_data_loader(batch_size=8)

        start = time.time()
        for train_loader_dict in train_generator:
            input_image = train_loader_dict['input'] # 入力画像を取得する
            target = train_loader_dict['real'] # 正解画像を取得する
            # 学習を1ステップ進める
            train_step(input_image, target, step, epoch)
            step = step + 1

            if step % 10 == 0:
                # ノートブックのセルの出力をクリアする
                display.clear_output(wait=True)
                # # 画像を学習経過中に保存する
                # generate_images(generator, example_input, target)
        start = time.time()
        
        # 10エポックごとにモデルのチェックポイントを保存する
        if epoch % 10 == 0:
            checkpoint_path = "checkpoints/cp-" + str(epoch) + ".h5"
            generator.save_weights(checkpoint_path)

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


# ジェネレータのロスの計算部分
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

# ディスクリミネーターのネットワークの定義
def Discriminator(image_shape):
    initializer = tf.random_normal_initializer(0., 0.02)

    input_image = tf.keras.layers.Input(image_shape, name='input_image')
    target_image = tf.keras.layers.Input(image_shape, name='target_image')

    x = tf.keras.layers.concatenate([input_image, target_image])  # (batch_size, 256, 256, channels*2)

    # 入力画像のサイズが512x512の場合に層を追加する
    if image_shape == (512, 512, 1):
        x = downsample(CHANNEL, 4)(x)
    down1 = downsample(n_E1, kernel_size_E1, 1, DropOut_E1, MaxPooling_E1, BatchNorm_E1)(x)
    down2 = downsample(n_E2, kernel_size_E2, 1, DropOut_E2, MaxPooling_E2, BatchNorm_E2)(x)
    down3 = downsample(n_E3, kernel_size_E3, stride_E3, DropOut_E3, MaxPooling_E3, BatchNorm_E3)(down2)
    down4 = downsample(n_E4, kernel_size_E4, stride_E4, DropOut_E4, MaxPooling_E4, BatchNorm_E4)(down3)
    down5 = downsample(n_E5, kernel_size_E5, stride_E5, DropOut_E5, MaxPooling_E5, BatchNorm_E5)(down4)
    down6 = downsample(n_E6, kernel_size_E6, stride_E6, DropOut_E6, MaxPooling_E6, BatchNorm_E6)(down5)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same',
                                     kernel_initializer=initializer)  # (batch_size, 30, 30, 1)

    x = last(down6)
    return tf.keras.Model(inputs=[input_image, target_image], outputs=x)


def discriminator_loss(disc_real_output, disc_generated_output):
    # 入力画像と正解画像のロス①
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    # 入力画像の生成画像のロス②
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    # ①と②とのトータルロス
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

# 学習に使うオプティマイザー
generator_optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) 
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    if CHANNEL == 1:
        plt.imshow(display_list[i].numpy().flatten().reshape(256, 256) * 0.5 + 0.5, cmap='gray') # モノクロ画像の場合
    else:
        plt.imshow(display_list[i]) # カラー画像の場合
    plt.axis('off')
  plt.show()



train_images = glob.glob('bottle/train/good/*')
train = []
for im in train_images:
    image = cv2.imread(im)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    train.append(image)

train = np.array(train)
train = train.astype('float32') /255.


PATH = 'bottle/train/good'
# カラーのチャンネル数を指定、カラー:3、モノクロ1を指定
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

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

generator = Generator(G_input_dim)
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator = Discriminator(G_input_dim)

fit()