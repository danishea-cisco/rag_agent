from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import json

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("./data/Profile.pdf")
text, _, images = text_from_rendered(rendered)

with open("data/output_text.md", "w", encoding="utf-8") as f:
    f.write(text)

with open("data/output_images.json", "w", encoding="utf-8") as f:
    json.dump(images, f, ensure_ascii=False, indent=2)
