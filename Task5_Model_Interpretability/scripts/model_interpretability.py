import shap
import lime
import lime.lime_text
from transformers import AutoModelForTokenClassification, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = AutoModelForTokenClassification.from_pretrained("./Task3_Model_FineTuning/saved_model")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Sample sentence for interpretation
sample_text = "ምን ይመስላል በ100 ብር ተሸምተው ስልኩን ግባ"
tokens = tokenizer.tokenize(sample_text)

# SHAP - Model Explainer
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer([sample_text])

# Visualize SHAP
shap.initjs()
shap.force_plot(shap_values[0])

# LIME - Text Explainer
class NERTextClassifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict_proba(self, texts):
        # Tokenize and encode the input text
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        return np.array(outputs.logits.detach())

# Create an explainer using LIME
lime_explainer = lime.lime_text.LimeTextExplainer(class_names=["B-Product", "B-LOC", "B-PRICE", "O"])
ner_classifier = NERTextClassifier(model, tokenizer)

# Explain the prediction
lime_exp = lime_explainer.explain_instance(sample_text, ner_classifier.predict_proba, num_features=10)

# Show LIME explanation
lime_exp.show_in_notebook(text=True)

# Show SHAP and LIME plots
plt.show()

