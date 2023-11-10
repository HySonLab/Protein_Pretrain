import transformers

# Define the model token for the pretrained model
model_token = "facebook/esm2_t30_150M_UR50D"

# Load the pretrained model for Masked Language Modeling (MLM)
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)

# Load the tokenizer associated with the pretrained model
esm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)
