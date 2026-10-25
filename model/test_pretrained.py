from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# ========================================
# LOAD PRETRAINED BART MODEL
# ========================================

print("Loading summarization model...")

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")

# ========================================
# ARTICLE
# ========================================

article = """
The Indian government has announced a new initiative to promote
renewable energy across the country. The program will support solar
and wind power projects and provide financial assistance to companies
developing clean energy technologies. Officials said the initiative
will help reduce carbon emissions and increase India's renewable
energy capacity over the coming years.
"""

reference_summary = """
The Indian government launched an initiative to promote solar and
wind energy, support clean technology companies, reduce carbon
emissions, and increase renewable energy capacity.
"""

# ========================================
# TOKENIZE
# ========================================

inputs = tokenizer(
    article,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

# ========================================
# GENERATE SUMMARY
# ========================================

outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    min_new_tokens=15,
    num_beams=4,
    do_sample=False
)

generated_summary = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# ========================================
# ROUGE
# ========================================

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = scorer.score(
    reference_summary,
    generated_summary
)

# ========================================
# DISPLAY
# ========================================

print("\n================================")
print("ORIGINAL ARTICLE")
print("================================")
print(article.strip())

print("\n================================")
print("REFERENCE SUMMARY")
print("================================")
print(reference_summary.strip())

print("\n================================")
print("GENERATED SUMMARY")
print("================================")
print(generated_summary)

print("\n================================")
print("ROUGE RESULTS")
print("================================")
print("ROUGE-1:", round(scores["rouge1"].fmeasure, 4))
print("ROUGE-2:", round(scores["rouge2"].fmeasure, 4))
print("ROUGE-L:", round(scores["rougeL"].fmeasure, 4))

# ========================================
# STATISTICS
# ========================================

original_words = len(article.split())
generated_words = len(generated_summary.split())

compression_ratio = (
    (1 - generated_words / original_words) * 100
)

print("\n================================")
print("SUMMARY STATISTICS")
print("================================")
print("Original words:", original_words)
print("Generated words:", generated_words)
print(
    "Compression ratio:",
    round(compression_ratio, 2),
    "%"
)

print("\n================================")
print("TEST COMPLETED")
print("================================")