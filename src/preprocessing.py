from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, AcceleratorDevice, AcceleratorOptions, PictureDescriptionApiOptions
from docling.datamodel.pipeline_options import EasyOcrOptions  
import re
import logging
from pathlib import Path
from config import CONFIG
import time
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
   level=logging.INFO,
   format="%(levelname)s - %(asctime)s - %(message)s"
   )
logger = logging.getLogger(__name__)

def picture_description(model: str = "gpt-4.1-nano", max_tokens: int = 200, temperature: float = 0.3):
      
      try:
         picture_desc_option = PictureDescriptionApiOptions(    
            url="https://api.openai.com/v1/chat/completions",
            headers={
               "Authorization": f"Bearer {CONFIG.OPENAI_API_KEY}",
               "Content-Type": "application/json",
            },
            params=dict(model=model, max_tokens=max_tokens, temperature=temperature),
            timeout=90,
            prompt="Describe this image in few sentences in a single paragraph"
         )
         
         return picture_desc_option

      except Exception as e:
         logger.error(f"Failed to create PictureDescriptionApiOptions: {str(e)}")
         return None

def pdf_pipeline_options(picture_desc_option, cuda: bool = False):
     
   pipeline_options = PdfPipelineOptions(
      do_ocr=True,
      do_table_structure=True,
      do_formula_enrichment=True,
      do_picture_description=True,
      picture_description_options=picture_desc_option,
      generate_picture_images=True,
      generate_page_images=True,
      images_scale=2,
      enable_remote_services=True,
      # table_structure_options={"do_cell_matching": True},
      ocr_options=EasyOcrOptions(lang=["en"]),
      accelerator_options=AcceleratorOptions(num_threads=8,
                                             device=AcceleratorDevice.CUDA if cuda else AcceleratorDevice.CPU)
  )
   
   return pipeline_options

def convert_pdf_to_markdown(pipeline_options, pdf_path: str):
   format_options = {
      InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
   }

   converter = DocumentConverter(format_options=format_options)
   result = converter.convert(str(pdf_path))
   markdown_text = result.document.export_to_markdown(image_mode="embedded")
   return markdown_text


def remove_base64_images(markdown_text: str):
    pattern = r'!\[.*?\]\(data:image/png;base64,([A-Za-z0-9+/=\n]+)\)'
    cleaned_md = re.sub(pattern, "", markdown_text)
    return cleaned_md

def process_pdfs(input_path: Path, output_path: Path, model: str ="gpt-4-turbo", cuda: bool = False):
    
    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_path}")
    
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No pdf files available: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)

    picture_desc = picture_description(model=model)
    pipeline_options = pdf_pipeline_options(picture_desc_option=picture_desc, cuda=cuda)

    outputs = []

    for i, pdf_file in enumerate(pdf_files, start=1):
        logger.info(f"{i}/{len(pdf_files)} Processing: {pdf_file}")
        start_time = time.perf_counter()
        
        try:
            markdown_text = convert_pdf_to_markdown(pipeline_options=pipeline_options, pdf_path=pdf_file)
            cleaned_md_text = remove_base64_images(markdown_text=markdown_text)

            output_file = output_path / f"{pdf_file.stem}_numthreads.md"
            output_file.write_text(cleaned_md_text, encoding="utf-8")
            outputs.append(output_file)
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Saved {output_file.name} - time: {elapsed_time:.2f} sec\n")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            continue
        
    return outputs
        

if __name__ == "__main__":
    input_dir = Path("research-papers")
    """
    print(input_dir)
    print(f"Input dir: {input_dir.resolve()}")
    print(f"Exists? {input_dir.exists()}")

    if not input_dir.exists():
      raise ValueError(f"Input directory does not exist: {input_dir.resolve()}")
   """
    output_dir = Path("md-research-papers")

    outputs = process_pdfs(input_dir, output_dir, model="gpt-4.1-nano", cuda=False)
    print(len(outputs))

    