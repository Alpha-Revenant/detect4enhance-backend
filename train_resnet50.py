import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# =============== STEP 1: SET PATHS ===============
TRAIN_FRAMES_DIR = r"E:\DAiSEE\DAiSEE\Train Frames"
VAL_FRAMES_DIR = r"E:\DAiSEE\DAiSEE\Validation Frames"
LABELS_PATH = r"E:\DAiSEE\DAiSEE\Labels"
MODEL_PATH = "engagement_model.h5"  # Model Save Path

TRAIN_LABELS = os.path.join(LABELS_PATH, "train_frame_labels.csv")
VAL_LABELS = os.path.join(LABELS_PATH, "val_frame_labels.csv")

# =============== STEP 2: LOAD LABELS & FIX FILENAMES ===============
def load_labels(csv_path, frames_dir):
    """Loads labels, appends .jpg extension, ensures valid paths & normalizes labels."""
    df = pd.read_csv(csv_path)

    # Rename 'Frame' column to 'filename'
    if "Frame" in df.columns:
        df.rename(columns={"Frame": "filename"}, inplace=True)

    # Ensure correct .jpg extension
    df["filename"] = df["filename"].astype(str).apply(lambda x: f"{x}.jpg" if not x.endswith(".jpg") else x)

    # Convert filename to full path
    df["filename"] = df["filename"].apply(lambda x: os.path.join(frames_dir, x))

    # Verify if files exist
    df = df[df["filename"].apply(os.path.exists)]  # Remove invalid entries

    # Normalize labels to range 0-1
    for col in ["Engagement", "Confusion", "Boredom", "Frustration"]:
        df[col] = df[col] / df[col].max()

    return df

train_labels = load_labels(TRAIN_LABELS, TRAIN_FRAMES_DIR)
val_labels = load_labels(VAL_LABELS, VAL_FRAMES_DIR)

# Check if images are found
if len(train_labels) == 0 or len(val_labels) == 0:
    print("❌ No valid images found! Check Train Frames & Validation Frames directories.")
    exit()

print(f"✅ Loaded {len(train_labels)} training images, {len(val_labels)} validation images.")

# =============== STEP 3: PREPARE IMAGE DATA ===============
datagen = ImageDataGenerator(rescale=1./255)

# Train Data
train_generator = datagen.flow_from_dataframe(
    dataframe=train_labels,
    x_col="filename",
    y_col=["Engagement", "Confusion", "Boredom", "Frustration"],
    target_size=(224, 224),
    batch_size=32,
    class_mode="raw",  # ✅ For multi-label classification
    shuffle=True
)

# Validation Data
val_generator = datagen.flow_from_dataframe(
    dataframe=val_labels,
    x_col="filename",
    y_col=["Engagement", "Confusion", "Boredom", "Frustration"],
    target_size=(224, 224),
    batch_size=32,
    class_mode="raw",  # ✅ For multi-label classification
    shuffle=True
)

# =============== STEP 4: LOAD OR INITIALIZE MODEL ===============
if os.path.exists(MODEL_PATH):
    print("✅ Found existing model, resuming training...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 No saved model found. Initializing new ResNet50 model...")

    # Load pre-trained ResNet50
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)  # Prevent overfitting
    x = Dense(4, activation="sigmoid")(x)  # ✅ Change from softmax → sigmoid

    # Define model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",  # ✅ Change from sparse_categorical_crossentropy → binary_crossentropy
        metrics=["accuracy"]
    )

print("✅ Model ready!")

# =============== STEP 5: TRAIN THE MODEL ===============
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

try:
    print("🚀 Training Started...")
    
    # **Print label format to verify correctness**
    print("🔍 Sample labels from train generator:")
    sample_labels = train_generator.labels[:5]
    print(sample_labels)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stop]
    )
    print("🎉 Training Complete!")

except Exception as e:
    print(f"❌ Error during training: {str(e)}")

# =============== STEP 6: SAVE THE FINAL MODEL ===============
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
