from CNNClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None  # Initialize the model as None to avoid potential reusability issues

    def get_base_model(self):
        # Load the base model from the specified path
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        # Recompile the model with a new optimizer instance to avoid variable mismatch issues
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        # Data augmentation and rescaling configuration for training and validation data
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Train data generator with optional augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save the model in the Keras recommended format to avoid warnings
        model.save(path, save_format='keras')

    def train(self, callback_list: list):
        # Calculate steps per epoch based on training and validation data
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model with the specified parameters
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the trained model to the specified path
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
