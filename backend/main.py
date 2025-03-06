# News Article Misleading Check
# This script is intended to eventually be able to read through a news article
# and its title and tell the user whether the articles content is misleading
# Author: Obeida Iqbal

from transformers import pipeline


def check_sentiment(prompts):
    """ Takes a list of strings as a parameter and prints the sentiment of each string """
    classifier = pipeline("sentiment-analysis")
    results = classifier(prompts)
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

def main():
    """ Main program """
    list = ["We are very happy to show you the 🤗 Transformers library.", "I hate driving so much", "I love that I hate you"]
    check_sentiment(list)

if __name__ == "__main__":
    main()