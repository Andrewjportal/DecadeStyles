import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix




nb_train_samples = 437
nb_validation_samples = 75
nb_classes = 5  # number of classes
epochs =10
BATCH_SIZE = 32


IMAGE_SIZE = 244


TRAINING_DIR = r'/Users/andrewportal/ai/hooked/Images'


data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.15,horizontal_flip=True)


train_gen = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")

validation_gen = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")


IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

neural_net = tf.keras.applications.vgg16.VGG16(input_shape=IMAGE_SHAPE, weights='imagenet', include_top=False)
neural_net.trainable=False


filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model = tf.keras.Sequential([
    neural_net,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(.75),
    tf.keras.layers.Dense(5,activation=tf.nn.softmax)])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit_generator(
    train_gen,
    steps_per_epoch=nb_train_samples // BATCH_SIZE,
    epochs=epochs,
    validation_data=validation_gen,
    validation_steps=nb_validation_samples // BATCH_SIZE,
    callbacks=callbacks_list)


model.save('hook_vgg16.model')
print("model saved")

Y_pred = model.predict_generator(validation_gen, nb_validation_samples // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')

c_matrix=confusion_matrix(validation_gen.classes, y_pred)

print(c_matrix)

np.save(confusion_matrix,c_matrix)

print(classification_report(validation_gen.classes, y_pred, target_names=CATEGORIES))