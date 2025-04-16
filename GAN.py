import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

class cycleGAN:
    def __init__(self, images_path, paintings_path):
        self.images_path = images_path
        self.paintings_path = paintings_path
        
        # Define loss object
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Define optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Build models
        self.generator = self.build_generator()  
        self.discriminator = self.build_discriminator() 
        
        self.splitting_loading()
    
    def load_image(self, img_path):
        img_path = img_path.numpy().decode('utf-8')
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        
        # Data augmentation
        img_array = tf.image.random_flip_left_right(img_array)  
        img_array = tf.image.random_brightness(img_array, max_delta=0.1) 

        return img_array
    
    def load_images_from_folder(self, folder):
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  
                yield img_path 

    def create_dataset(self, folder):
        image_paths = list(self.load_images_from_folder(folder))
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: tf.py_function(func=self.load_image, inp=[x], Tout=tf.float32))
        return dataset

    def splitting_loading(self):
        images_folder = self.images_path
        paintings_folder = self.paintings_path
        
        # Create datasets
        self.train_images = self.create_dataset(images_folder).batch(8)  
        self.train_paintings = self.create_dataset(paintings_folder).batch(8) 

    def residual_block(self, x):
        """A residual block for the generator."""
        res = layers.Conv2D(256, kernel_size=3, padding='same')(x)
        res = layers.BatchNormalization()(res)
        res = layers.ReLU()(res)
        res = layers.Conv2D(256, kernel_size=3, padding='same')(res)
        res = layers.BatchNormalization()(res)
        return layers.add([x, res])
    
    def build_generator(self):
        """Builds the generator model."""
        inputs = layers.Input(shape=(256, 256, 3))  # Input shape (height, width, channels)
        
        # Downsampling
        x = layers.Conv2D(64, kernel_size=7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Residual Blocks
        for _ in range(9):  # Number of residual blocks
            x = self.residual_block(x)
        
        # Upsampling
        x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv2D(3, kernel_size=7, padding='same', activation='tanh')(x)  # Output layer
        
        return Model(inputs, outputs)
    
    def build_discriminator(self):
        """Builds the discriminator model."""
        inputs = layers.Input(shape=(256, 256, 3))  # Input shape (height, width, channels)
        
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(1, kernel_size=4, padding='same')(x)  # Output layer
        outputs = layers.Activation('sigmoid')(x)  # Sigmoid activation for binary classification
        
        return Model(inputs, outputs)

    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            # Generate images
            fake_y = self.generator(real_x, training=True)
            cycled_x = self.generator(fake_y, training=True)

            # Discriminator predictions
            disc_real_x = self.discriminator(real_x, training=True)
            disc_fake_y = self.discriminator(fake_y, training=True)

            # Calculate losses
            gen_loss = self.loss_object(tf.ones_like(disc_fake_y), disc_fake_y)
            disc_loss = self.loss_object(tf.ones_like(disc_real_x), disc_real_x) + \
                        self.loss_object(tf.zeros_like(disc_fake_y), disc_fake_y)

            # Cycle consistency loss
            cycle_loss = tf.reduce_mean(tf.abs(real_x - cycled_x))

            # Total generator loss
            total_gen_loss = gen_loss + (10 * cycle_loss)  # Weighting cycle loss

        # Calculate gradients
        generator_gradients = tape.gradient(total_gen_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    def running(self, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            for real_x, real_y in zip(self.train_images, self.train_paintings):
                self.train_step(real_x, real_y)

            print(f"Completed Epoch {epoch+1}/{epochs}")

        # Save the models after training
        self.save_models()
        
    def convert_image_to_painting(self, image_path):
        # Load the input image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Use the generator model to translate the image into a painting
        painting = self.generator.predict(img_array[np.newaxis, ...])
        
        # Convert the painting back to the range [0, 1]
        painting = (painting + 1) / 2
        
        return painting

    def save_models(self):
        self.generator.save(r'E:\DRDO\Notebooks\model\generator_model.keras')  
        self.discriminator.save(r'E:\DRDO\Notebooks\model\discriminator_model.keras')  

        # Save the optimizer states
        with open(r'E:\DRDO\Notebooks\model\optimizer_states.pkl', 'wb') as f:
            pickle.dump({
                'generator_optimizer': self.generator_optimizer.variables,
                'discriminator_optimizer': self.discriminator_optimizer.variables
            }, f)


    def load_models(self):
        # Load models
        self.generator = load_model(r'E:\DRDO\Notebooks\model\generator_model.keras') 
        self.discriminator = load_model(r'E:\DRDO\Notebooks\model\discriminator_model.keras')

        # Reinitialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Build optimizer variables
        self.generator_optimizer.build(self.generator.trainable_variables)
        self.discriminator_optimizer.build(self.discriminator.trainable_variables)

        # Load optimizer states
        with open(r'E:\DRDO\Notebooks\model\optimizer_states.pkl', 'rb') as f:
            optimizer_states = pickle.load(f)

            for var, saved_var in zip(self.generator_optimizer.variables, optimizer_states['generator_optimizer']):
                var.assign(saved_var)

            for var, saved_var in zip(self.discriminator_optimizer.variables, optimizer_states['discriminator_optimizer']):
                var.assign(saved_var)

