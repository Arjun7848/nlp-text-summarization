from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_NAME = "facebook/bart-large-cnn"

print("Loading summarization model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        summary=None,
        original_words=0,
        summary_words=0,
        compression=0,
        rouge1=None,
        rouge2=None,
        rougel=None,
        article="",
        reference_summary=""
    )


# ============================================================
# SUMMARIZE
# ============================================================

@app.route("/summarize", methods=["POST"])
def summarize():

    article = request.form.get("article", "").strip()
    reference_summary = request.form.get(
        "reference_summary", ""
    ).strip()

    # Check empty article
    if not article:
        return render_template(
            "index.html",
            summary="Please enter an article.",
            original_words=0,
            summary_words=0,
            compression=0,
            rouge1=None,
            rouge2=None,
            rougel=None,
            article=article,
            reference_summary=reference_summary
        )

    # Count original words
    original_words = len(article.split())

    print("Received article.")
    print("Original words:", original_words)

    # ========================================================
    # TOKENIZATION
    # ========================================================

    inputs = tokenizer(
        article,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    print("Generating summary...")

    # ========================================================
    # GENERATE SUMMARY
    # ========================================================

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        min_new_tokens=15,
        num_beams=4,
        do_sample=False
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    print("Summary generated successfully.")

    # ========================================================
    # SUMMARY WORD COUNT
    # ========================================================

    summary_words = len(summary.split())

    # ========================================================
    # COMPRESSION
    # ========================================================

    compression = round(
        (1 - summary_words / original_words) * 100,
        2
    )

    # ========================================================
    # ROUGE
    # ========================================================

    rouge1 = None
    rouge2 = None
    rougel = None

    if reference_summary:

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        scores = scorer.score(
            reference_summary,
            summary
        )

        rouge1 = round(
            scores["rouge1"].fmeasure,
            4
        )

        rouge2 = round(
            scores["rouge2"].fmeasure,
            4
        )

        rougel = round(
            scores["rougeL"].fmeasure,
            4
        )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return render_template(
        "index.html",
        summary=summary,
        original_words=original_words,
        summary_words=summary_words,
        compression=compression,
        rouge1=rouge1,
        rouge2=rouge2,
        rougel=rougel,
        article=article,
        reference_summary=reference_summary
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("Starting Text Summarization Web Application...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )