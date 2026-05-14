#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Script for Hand Gesture Recognition
Builds and trains a lightweight CNN model optimized for Raspberry Pi 5
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_DIR = 'images'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

# Model files
MODEL_H5 = 'model.h5'
MODEL_TFLITE = 'gesture_model.tflite'
LABELS_JSON = 'gesture_labels.json'

def create_lightweight_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=5):
    """
    Create a very lightweight CNN model optimized for edge devices
    Alternative: Use MobileNetV2 (commented out below)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth conv block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=5):
    """
    Alternative: Use MobileNetV2 (lighter but requires more memory)
    Uncomment this function and use it instead of create_lightweight_model if preferred
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.35  # Width multiplier (0.35 = 35% of original width, very light)
    )
    
    # Freeze base model initially (optional - can fine-tune later)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model


def save_label_mapping(class_labels, output_path=LABELS_JSON):
    payload = {
        "labels": class_labels,
        "num_classes": len(class_labels),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved labels mapping to '{output_path}'")


def prepare_data_generators(dataset_dir, img_size, batch_size, validation_split=0.2):
    """Prepare data generators with augmentation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=validation_split
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, model_type='mobilenet'):
    """Main training function"""
    print("="*60)
    print("HAND GESTURE RECOGNITION - MODEL TRAINING")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        print("Please run collect_data.py first to collect images.")
        return

    class_folders = sorted(
        [entry.name for entry in os.scandir(dataset_dir) if entry.is_dir()]
    )
    if not class_folders:
        print(f"Error: No class folders found under '{dataset_dir}'.")
        return
    print(f"Discovered classes ({len(class_folders)}): {class_folders}")
    
    # Prepare data generators
    print("\n[1/5] Preparing data generators...")
    train_gen, val_gen = prepare_data_generators(
        dataset_dir=dataset_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    
    print(f"Found {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    class_labels = [label for label, _ in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]
    save_label_mapping(class_labels)

    # Create model
    print("\n[2/5] Creating model...")
    num_classes = train_gen.num_classes
    if model_type == 'lightweight':
        model = create_lightweight_model(
            input_shape=(img_size, img_size, 3),
            num_classes=num_classes,
        )
        base_model = None
    else:
        model, base_model = create_mobilenet_model(
            input_shape=(img_size, img_size, 3),
            num_classes=num_classes,
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Class weighting for imbalanced datasets
    y_train = train_gen.classes
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    print(f"Class weights: {class_weight}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_H5,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
    ]

    # Train model
    print("\n[3/5] Training model...")

    if base_model is None:
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
    else:
        warmup_epochs = max(4, epochs // 3)
        finetune_epochs = max(4, epochs - warmup_epochs)

        print(f"Warmup phase: {warmup_epochs} epochs (frozen backbone)")
        warmup_history = model.fit(
            train_gen,
            epochs=warmup_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        print(f"Fine-tune phase: {finetune_epochs} epochs (top backbone layers unfrozen)")
        base_model.trainable = True
        freeze_until = max(0, len(base_model.layers) - 40)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        finetune_history = model.fit(
            train_gen,
            epochs=finetune_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        # Merge histories so plotting remains unchanged.
        history = warmup_history
        history.history['accuracy'].extend(finetune_history.history.get('accuracy', []))
        history.history['val_accuracy'].extend(finetune_history.history.get('val_accuracy', []))
        history.history['loss'].extend(finetune_history.history.get('loss', []))
        history.history['val_loss'].extend(finetune_history.history.get('val_loss', []))
    
    # Load best model
    print("\n[4/5] Loading best model...")
    model = keras.models.load_model(MODEL_H5)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=1)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Generate predictions for confusion matrix
    print("\n[5/5] Generating predictions and metrics...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels,
        zero_division=0,
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history saved as 'training_history.png'")
    
    # Convert to TFLite
    print("\n" + "="*60)
    print("CONVERTING TO TENSORFLOW LITE")
    print("="*60)
    convert_to_tflite(model)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved as: {MODEL_H5}")
    print(f"TFLite model saved as: {MODEL_TFLITE}")


def convert_to_tflite(model):
    """Convert Keras model to TensorFlow Lite with quantization"""
    
    # Convert to TFLite (float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save float32 model
    with open(MODEL_TFLITE, 'wb') as f:
        f.write(tflite_model)
    
    float32_size = len(tflite_model) / 1024
    print(f"Float32 TFLite model size: {float32_size:.2f} KB")
    
    # Convert with dynamic range quantization (int8)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    
    # Save quantized model
    with open(MODEL_TFLITE, 'wb') as f:
        f.write(tflite_quantized_model)
    
    quantized_size = len(tflite_quantized_model) / 1024
    print(f"Quantized (int8) TFLite model size: {quantized_size:.2f} KB")
    print(f"Size reduction: {(1 - quantized_size/float32_size)*100:.1f}%")
    
    return tflite_quantized_model


if __name__ == '__main__':
    # Set memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    parser = argparse.ArgumentParser(description='Train CNN gesture model from image folders')
    parser.add_argument('--dataset-dir', default=DATASET_DIR, help='Path to dataset root (class subfolders)')
    parser.add_argument('--img-size', type=int, default=IMG_SIZE, help='Square image size for training')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--model-type', choices=['lightweight', 'mobilenet'], default='mobilenet', help='Model architecture to train')
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        model_type=args.model_type,
    )

