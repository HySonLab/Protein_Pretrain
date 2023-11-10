import transformers

model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)

