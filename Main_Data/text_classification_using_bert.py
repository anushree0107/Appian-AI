import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and display sample of data
data = pd.read_csv(r'C:\Users\Anushree\Downloads\Appian AI\AppianChallenge\SyntheticDataGeneration\Main_Data\Final_data.csv', encoding='latin-1')
print("Sample data:")
print(data.head())
print("\nUnique labels:", data['label'].unique())
print("Label distribution:\n", data['label'].value_counts())

# Split into train/test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Label encoding
label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['label'])  # Changed from 'Sentiment' to 'label'
y = to_categorical(y)
y_test = label.transform(test_data['label'])  # Changed from 'Sentiment' to 'label'
y_test_cat = to_categorical(y_test)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 256  # Increased max_length since we might have longer texts

# Tokenize and encode sequences
def encode_text(texts):
    # Convert texts to list and ensure all elements are strings
    if isinstance(texts, pd.Series):
        texts = texts.astype(str).tolist()
    elif isinstance(texts, list):
        texts = [str(text) for text in texts]
    
    tokens = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='tf'
    )
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask']
    }

# Encode train and test texts
train_encodings = encode_text(train_data['text'])  # Changed from 'Text' to 'text'
test_encodings = encode_text(test_data['text'])    # Changed from 'Text' to 'text'

# Create BERT model
def create_bert_model(num_classes):
    # Load BERT model
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    
    # Input layers
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    # BERT layer - convert inputs to tensors
    bert_outputs = bert(
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    )
    pooled_output = bert_outputs[1]
    
    # Add multiple dense layers with dropout for better classification
    x = Dense(512, activation='relu')(pooled_output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model

# Create and compile model
num_classes = len(label.classes_)
model = create_bert_model(num_classes)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
checkpoint = ModelCheckpoint(
    'best_document_classifier.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True
)

earlystopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    },
    y,
    validation_data=(
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        y_test_cat
    ),
    epochs=5,  # Increased epochs
    batch_size=16,  # Reduced batch size for better learning
    callbacks=[checkpoint, earlystopping],
    verbose=1
)

# Evaluate model
y_pred = model.predict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask']
})
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label.classes_))

# # Plot confusion matrix
# plt.figure(figsize=(12,10))
# cm = confusion_matrix(y_test, y_pred_classes)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=label.classes_,
#             yticklabels=label.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Plot training history
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Print example predictions
def predict_category(text):
    # Encode the text
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    
    # Make prediction
    prediction = model.predict({
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    })
    
    # Get predicted class
    predicted_class = label.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Test some examples
print("\nExample Predictions:")
test_samples = test_data.sample(5)
for _, row in test_samples.iterrows():
    predicted_class, confidence = predict_category(row['text'])
    print(f"\nText: {row['text'][:100]}...")
    print(f"True Label: {row['label']}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2f}") 