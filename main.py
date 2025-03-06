# News Article Misleading Check
# This script is intended to read through a news article and its title
# and tell the user whether the article is misleading
# Author: Obeida Iqbal

from transformers import pipeline


def main():
    """ Main program """
    classifier = pipeline("sentiment-analysis")
    results = classifier(["We are very happy to show you the 🤗 Transformers library.", "I hate driving so much", "I love that I hate you"])
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

if __name__ == "__main__":
    main()