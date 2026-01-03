from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import json

def main():

    PDF_FILE = "Agentic_Design_Patterns"

    converter = PdfConverter(
        artifact_dict=create_model_dict(device="mps"),
    )
    rendered = converter(f"./data/input/{PDF_FILE}.pdf")
    text, _, images = text_from_rendered(rendered)

    with open(f"data/output/{PDF_FILE}_text.md", "w", encoding="utf-8") as f:
        f.write(text)

    with open("data/{PDF_FILE}_images.json", "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
