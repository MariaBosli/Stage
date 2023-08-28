import torch, os
import pandas as pd
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device

data = pd.read_csv('D:/articles.csv')
# Supprimer les colonnes inutiles
colonnes_inutiles = ['id', 'date', 'year', 'year', 'month', 'url']
data = data.drop(colonnes_inutiles, axis=1)

#supprimer les valeurs manquantes
data.dropna(inplace=True)

author_counts = data['author'].value_counts()
top_2_authors = author_counts.nlargest(2).index
filtered_data = data[data['author'].isin(top_2_authors)]

# Supprimer les colonnes inutiles
colonnes_inutiles = ['title', 'publication']
filtered_data = filtered_data.drop(colonnes_inutiles, axis=1)

labels = filtered_data['author'].unique().tolist()
labels = [s.strip() for s in labels ]
labels

for key, value in enumerate(labels):
    print(value)

NUM_LABELS= len(labels)

id2label={id:label for id,label in enumerate(labels)}

label2id={label:id for id,label in enumerate(labels)}

label2id
id2label

filtered_data["labels"]=filtered_data.author.map(lambda x: label2id[x.strip()])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)

tokenizer.save_pretrained(save_directory)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(device)

SIZE= filtered_data.shape[0]

train_texts= list(filtered_data.content[:SIZE//2])

val_texts=   list(filtered_data.content[SIZE//2:(3*SIZE)//4 ])

test_texts=  list(filtered_data.content[(3*SIZE)//4:])

train_labels= list(filtered_data.labels[:SIZE//2])

val_labels=   list(filtered_data.labels[SIZE//2:(3*SIZE)//4])

test_labels=  list(filtered_data.labels[(3*SIZE)//4:])

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class DataLoader(Dataset):
    """
    Custom Dataset class for handling tokenized text data and corresponding labels.
    Inherits from torch.utils.data.Dataset.
    """
    def __init__(self, encodings, labels):
        """
        Initializes the DataLoader class with encodings and labels.

        Args:
            encodings (dict): A dictionary containing tokenized input text data
                              (e.g., 'input_ids', 'token_type_ids', 'attention_mask').
            labels (list): A list of integer labels for the input text data.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized data and the corresponding label for a given index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            item (dict): A dictionary containing the tokenized data and the corresponding label.
        """
        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(self.labels)
    



    
train_dataloader = DataLoader(train_encodings, train_labels)

val_dataloader = DataLoader(val_encodings, val_labels)

test_dataset = DataLoader(test_encodings, test_labels)


from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.
    
    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of 
              that observation belonging to a certain class.
              
    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    # Extract true labels from the input object
    labels = pred.label_ids
    
    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)
    
    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)
    
    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./TTC4900Model', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory                 
    logging_dir='./multi-class-logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps", 
    fp16=False,
    load_best_model_at_end=True
)

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,                 
    train_dataset=train_dataloader,         
    eval_dataset=val_dataloader,            
    compute_metrics= compute_metrics
)

trainer.train()

save_directory = "D:\model" 

model.save_pretrained(save_directory)

trainer.evaluate()

model_path = "D:\modelvf"


model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
nlp= pipeline("text-classification", model=model, tokenizer=tokenizer)

nlp("unday on CBS’s “Face The Nation,” while discussing the Republicans the repeal and replacement of the Affordable Care Act, Sen. Rand Paul ( ) said there is a “separation between” Speaker of the House Paul Ryan’s ( ) and President Donald Trump.  Paul said, “I think there is a separation between the two. I have talked to the president, I think three times on Obamacare and I hear from him he is willing to negotiate. You know what I hear from Paul Ryan ‘it is a binary choice, young man.’ But what is a binary choice, his way or the highway?” Follow Pam Key on Twitter @pamkeyNEN")