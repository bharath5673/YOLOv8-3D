import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity

import tensorflow as tf
print('\ntensorflow version : ',tf.__version__)
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


from tensorflow.keras.applications import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Loss


from tensorflow.keras import backend as K
from IPython.display import SVG
import matplotlib.pyplot as plt

import copy
import cv2, os
import numpy as np
from random import shuffle
import pandas as pd

import tqdm
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"



BIN, OVERLAP = 6, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
BATCH_SIZE = 8
AUGMENTATION = False

## select model  and input size
# select_model = 'resnet18'
# select_model = 'resnet50'
# select_model ='resnet101'
# select_model = 'resnet152'
# select_model = 'vgg11'
# select_model = 'vgg16'
# select_model = 'vgg19'
# select_model = 'efficientnetb0'
# select_model = 'efficientnetb5'
select_model = 'mobilenetv2'




label_dir = '/home/bharath/Downloads/test_codes/3Dbbox/kitti/training/label_2/'
image_dir = '/home/bharath/Downloads/test_codes/3Dbbox/kitti/training/image_2/'



seq = iaa.Sequential([
    iaa.Crop(px=(0, 7)),  # will randomly crop between 0 to 7 pixels from the left side.
    iaa.Crop(px=(7, 0)),  # will randomly crop between 0 to 7 pixels from right to left
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0))  
])



###### preProcessing #####

def compute_anchors(angle):
    anchors = []
    wedge = 2.*np.pi/BIN

    l_index = int(angle/wedge)
    r_index = l_index + 1
    
    if (angle - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
        anchors.append([l_index, angle - l_index*wedge])
    if (r_index*wedge - angle) < wedge/2 * (1+OVERLAP/2):
        anchors.append([r_index%BIN, angle - r_index*wedge])
    return anchors



def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key:np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key:0 for key in VEHICLES}
        
    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name':line[0],
                       'image':image_file,
                       'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       'dims':np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)
            
    return all_objs, dims_avg

all_objs, dims_avg = parse_annotation(label_dir, image_dir)

for obj in all_objs:
    # Fix dimensions
    obj['dims'] = obj['dims'] - dims_avg[obj['name']]
    
    # Fix orientation and confidence for no flip
    orientation = np.zeros((BIN,2))
    confidence = np.zeros(BIN)
    
    anchors = compute_anchors(obj['new_alpha'])
    
    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1.
        
    confidence = confidence / np.sum(confidence)
        
    obj['orient'] = orientation
    obj['conf'] = confidence
        
    # Fix orientation and confidence for flip
    orientation = np.zeros((BIN,2))
    confidence = np.zeros(BIN)
    
    anchors = compute_anchors(2.*np.pi - obj['new_alpha'])
    
    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1
        
    confidence = confidence / np.sum(confidence)
        
    obj['orient_flipped'] = orientation
    obj['conf_flipped'] = confidence

def prepare_input_and_output(train_inst):
    ### Prepare image patch
    xmin = train_inst['xmin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = train_inst['ymin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = train_inst['xmax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = train_inst['ymax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    
    img = cv2.imread(image_dir + train_inst['image'])
    img = copy.deepcopy(img[ymin:ymax+1,xmin:xmax+1]).astype(np.float32)
    
    # re-color the image
    #img += np.random.randint(-2, 3, img.shape).astype('float32')
    #t  = [np.random.uniform()]
    #t += [np.random.uniform()]
    #t += [np.random.uniform()]
    #t = np.array(t)

    #img = img * (1 + t)
    #img = img / (255. * 2.)

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)
        
    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img - np.array([[[103.939, 116.779, 123.68]]])
    #img = img[:,:,::-1]
    
    ### Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


def augment_image(image):
    augmented_image = seq.augment_images([image])[0]
    return augmented_image


def data_gen(all_objs, batch_size):
    num_obj = len(all_objs)
    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        if not AUGMENTATION:
            x_batch = np.zeros((batch_size, 224, 224, 3))
            d_batch = np.zeros((batch_size, 3))
            o_batch = np.zeros((batch_size, BIN, 2))
            c_batch = np.zeros((batch_size, BIN))

            for idx, key in enumerate(keys[l_bound:r_bound]):
                #input image and fix object's orientation and confidence
                image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

                # Original images
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

            yield x_batch, [d_batch, o_batch, c_batch]

        if AUGMENTATION:
            x_batch = np.zeros((2 * batch_size, 224, 224, 3))
            d_batch = np.zeros((2 * batch_size, 3))
            o_batch = np.zeros((2 * batch_size, BIN, 2))
            c_batch = np.zeros((2 * batch_size, BIN))

            for idx, key in enumerate(keys[l_bound:r_bound]):
                # input image and fix object's orientation and confidence
                image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

                # Original images
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

                # Augmented images
                x_batch[idx + batch_size, :] = augment_image(image)
                d_batch[idx + batch_size, :] = dimension.copy()
                o_batch[idx + batch_size, :] = orientation.copy()
                c_batch[idx + batch_size, :] = confidence.copy()

            yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj:
            r_bound = num_obj

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)





######## Regression netwrok ######


input_shape = (224, 224, 3)

if select_model == 'resnet18':
    ARCH = ResNet18
if select_model == 'resnet50':
    ARCH = ResNet50
if select_model == 'resnet101':
    ARCH = ResNet101
if select_model == 'resnet152':
    ARCH = ResNet152
if select_model == 'vgg11':
    ARCH = VGG11
if select_model == 'vgg16':
    ARCH = VGG16
if select_model == 'vgg19':
    ARCH = VGG19
if select_model == 'efficientnetb0':
    ARCH = EfficientNetB0
if select_model == 'efficientnetb5':
    ARCH = EfficientNetB0
if select_model == 'mobilenetv2':
    ARCH = MobileNetV2



# Construct the network
base_model = ARCH(weights='imagenet', include_top=False, input_shape=input_shape)
# base_model = ARCH(weights=None, include_top=True, input_shape=input_shape)


# Add additional layers for orientation prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)

dimension   = Dense(512)(x)
dimension   = LeakyReLU(alpha=0.1)(dimension)
dimension   = Dropout(0.2)(dimension)
dimension   = Dense(3)(dimension)                                               
dimension   = LeakyReLU(alpha=0.1, name='dimension')(dimension)

orientation = Dense(256)(x)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Dropout(0.2)(orientation)
orientation = Dense(BIN*2)(orientation)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Reshape((BIN,-1))(orientation)                                    
orientation = Lambda(l2_normalize, name='orientation')(orientation)

confidence  = Dense(256)(x)
confidence  = LeakyReLU(alpha=0.1)(confidence)
confidence  = Dropout(0.2)(confidence)
confidence  = Dense(BIN, activation='softmax', name='confidence')(confidence)  

model = Model(inputs= base_model.input, outputs=[dimension, orientation, confidence])






###### Training ##########
@tf.keras.saving.register_keras_serializable()
def orientation_loss(y_true, y_pred):
    # Find number of anchors
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    
    # Define the loss
    loss = -(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    loss = tf.reduce_sum(loss, axis=1)
    epsilon = 1e-5  ##small epsilon value to prevent division by zero.
    anchors = anchors + epsilon
    loss = loss / anchors
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss 

    return loss





if __name__ == '__main__':
    if not os.path.exists(select_model):
        os.makedirs(select_model)
    early_stop  = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    checkpoint  = ModelCheckpoint('./'+select_model+'/'+select_model+'_weights.h5', monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir='./'+select_model+'/logs/', histogram_freq=0, write_graph=True, write_images=False)

    all_exams = len(all_objs) 
    trv_split  = int(0.85*all_exams)
    batch_size = BATCH_SIZE
    np.random.shuffle(all_objs)

    train_gen = data_gen(all_objs[:trv_split],          batch_size)
    valid_gen = data_gen(all_objs[trv_split:all_exams], batch_size)

    train_num = int(np.ceil(trv_split/batch_size))
    valid_num = int(np.ceil((all_exams - trv_split)/batch_size))
    print('train and val split : ', trv_split, all_exams - trv_split)



    # Check if the file 'weights.h5' exists
    if os.path.exists('./'+select_model+'/'+select_model+'_weights.h5'):
        # Load the model
        model = load_model('./'+select_model+'/'+select_model+'_weights.h5')
        print(model.summary())
        print('loading file ...'+select_model+'_weights.h5...!')
    else:
        print('The file '+select_model+'_weights.h5 does not exist ..starting form epoch 1.')



    # Try to read the history file
    try:
        history_df = pd.read_csv('./'+select_model+'/'+select_model+'_training_history.csv')
        last_epoch = history_df.index[-1] + 1  # Get the last epoch and add 1 for the next epoch
    except FileNotFoundError:
        last_epoch = 0  # If the file doesn't exist, start from epoch 1
    print(f'Last epoch number: {last_epoch}')


    # minimizer  = SGD(learning_rate=0.000001, momentum=0.8)
    minimizer = Adam(learning_rate=1e-5)
    model.compile(
                    optimizer=minimizer,
                    loss={
                        'dimension': 'mean_squared_error', 
                        # 'dimension': 'mean_absolute_error', 
                        # 'dimension': 'mean_squared_logarithmic_error', 
                        'orientation': orientation_loss, 
                        'confidence': 'binary_crossentropy',
                        # 'confidence': 'categorical_crossentropy',
                        },
                    loss_weights={'dimension': 5.0, 'orientation': 1.5, 'confidence': 0.5},
                    metrics={
                        'dimension': 'mse', 
                        'orientation': 'mse', 
                        'confidence': 'accuracy',
                        })

    history = model.fit(train_gen, 
                        initial_epoch=last_epoch,
                        steps_per_epoch = train_num, 
                        epochs = last_epoch + 10, 
                        verbose = 1, 
                        validation_data = valid_gen, 
                        validation_steps = valid_num, 
                        callbacks = [early_stop, checkpoint, tensorboard], 
                        shuffle=True,
                        )

    # d:0.0088 o:0.0042, c:0.0098    ### targetloss
    
    history_file = './'+select_model+'/'+select_model+'_training_history.csv'
    # Check if the history CSV file exists
    if os.path.exists(history_file):
        existing_history_df = pd.read_csv(history_file) # Load the existing history data
        # Concatenate the new history data with the existing data
        history_df = pd.concat([existing_history_df, pd.DataFrame(history.history)], ignore_index=True)
    else:
        history_df = pd.DataFrame(history.history) # If the file doesn't exist, create a new DataFrame

    # Save the updated history to CSV
    history_df.to_csv(history_file, index=False)


    # # Plot training & validation loss values
    # Read the training history CSV file
    history_df = pd.read_csv('./'+select_model+'/'+select_model+'_training_history.csv')

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training & validation loss values
    ax1.plot(history_df['loss'], label='Train Loss')
    ax1.plot(history_df['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Plot training & validation accuracy values
    ax2.plot(history_df['dimension_mse'], label='Dimensions Acc')
    ax2.plot(history_df['val_dimension_mse'], label='Validation Dimensions Acc')
    ax2.plot(history_df['orientation_mse'], label='Orientation Acc')
    ax2.plot(history_df['val_orientation_mse'], label='Validation Orientation Acc')
    ax2.plot(history_df['confidence_accuracy'], label='Confidence Acc')
    ax2.plot(history_df['val_confidence_accuracy'], label='Validation Confidence Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    # Adjust the layout
    plt.tight_layout()
    plt.suptitle(select_model)
    plt.savefig('./'+select_model+'/'+select_model+'_results_plot.png')
    # Show the plot
    # plt.show()

