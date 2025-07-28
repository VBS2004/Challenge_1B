import os
import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pdfplumber
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

class IntegratedPDFAnalyzer:
    """Integrated PDF analyzer that extracts outlines and performs similarity analysis."""
    
    def __init__(self, model_path: str = "./models/all-MiniLM-L6-v2", 
                 top_k: int = 5, max_pages: int = 50):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to local sentence transformer model (for offline use)
            top_k: Number of top sections to extract
            max_pages: Maximum pages to process per PDF
        """
        print(f"Loading model from: {model_path}")
        try:
            # Try to load local model first (offline mode)
            if os.path.exists(model_path):
                self.model = SentenceTransformer(model_path)
                print("✅ Loaded model from local path (offline mode)")
            else:
                # Fallback to downloading (online mode)
                print("⚠️ Local model not found, downloading from internet...")
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                print("✅ Downloaded model from internet")
        except Exception as e:
            raise Exception(f"Failed to load sentence transformer model: {e}")
        
        self.top_k = top_k
        self.max_pages = max_pages
    
    def normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'\W+', '', text.lower())
    
    def is_heading_like(self, line_text: str, level: str) -> bool:
        """Determine if a line looks like a heading."""
        text = line_text.strip()
        words = text.split()
        
        # At least 2 characters and must have a letter
        if len(text) < 2 or not re.search(r'[a-zA-Z]', text):
            return False
        
        # Filter out lines that are code, JSON, or mostly symbols/punctuation
        if all(ch in "•-o{}[]:,.\"'" for ch in text):
            return False
        if text.startswith("{") or text.startswith("[") or text.startswith("\"") or text.startswith("]"):
            return False
        if not (1 <= len(words) <= 14):
            return False
        if "http" in text or "www." in text:
            return False
        if re.match(r"^[\W\d]+$", text):
            return False
        if text.endswith(".") or text.endswith("?") or text.endswith("!"):
            return False
        if sum(text.count(ch) for ch in ",.;?!") > 2:
            return False
        if len(text) > 80:
            return False
        
        # For H3, only allow all caps or up to 4 words
        if level == "H3":
            if text.upper() == text and len(words) <= 14:
                return True
            if len(words) <= 4:
                return True
            return False
        
        return True
    
    def extract_pdf_outline(self, pdf_path: str) -> Dict[str, any]:
        """Extract title and detailed outline from a PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) > self.max_pages:
                    print(f"⚠️ Skipping {os.path.basename(pdf_path)} ({len(pdf.pages)} > {self.max_pages} pages)")
                    return {"title": "", "outline": []}
                
                # Get all font sizes to determine heading levels
                all_sizes = []
                for pg in pdf.pages:
                    for ch in pg.chars:
                        all_sizes.append(round(ch['size'], 1))
                
                uniq_sizes = sorted(set(all_sizes), reverse=True)
                if not uniq_sizes:
                    return {"title": "", "outline": []}
                
                # Map sizes to heading levels
                size_to_level = {}
                if len(uniq_sizes) >= 1:
                    size_to_level[uniq_sizes[0]] = "H1"
                if len(uniq_sizes) >= 2:
                    size_to_level[uniq_sizes[1]] = "H2"
                for sz in uniq_sizes[2:]:
                    size_to_level[sz] = "H3"
                if len(uniq_sizes) >= 3:
                    size_to_level[uniq_sizes[2]] = "H3"

                headings_with_text = []
                
                # Process each page
                for page_no, pg in enumerate(pdf.pages, start=1):
                    # Group characters by line
                    lines = {}
                    for ch in pg.chars:
                        top = round(ch['top'] / 2) * 2
                        lines.setdefault(top, []).append(ch)
                    
                    page_headings = []
                    
                    # Identify headings on this page
                    for top in sorted(lines):
                        line_chars = sorted(lines[top], key=lambda c: c['x0'])
                        line_text = ''.join(ch['text'] for ch in line_chars).strip()
                        if not line_text:
                            continue
                        
                        sizes = [round(c['size'], 1) for c in line_chars]
                        main_size = max(set(sizes), key=sizes.count)
                        level = size_to_level.get(main_size, "H3")
                        
                        if not self.is_heading_like(line_text, level):
                            continue
                        
                        heading_info = {
                            "level": level, 
                            "text": line_text, 
                            "page": page_no,
                            "top": top
                        }
                        page_headings.append(heading_info)
                    
                    # Extract content for each heading
                    page_text = pg.extract_text() or ""
                    page_lines = page_text.split('\n')
                    
                    for i, heading in enumerate(page_headings):
                        content_text = self._extract_heading_content(
                            heading, page_headings[i+1:], page_lines
                        )
                        
                        headings_with_text.append({
                            "level": heading["level"],
                            "text": heading["text"],
                            "page": heading["page"],
                            "content": content_text
                        })

                # Determine title (first H1 or first heading)
                title = ""
                for h in headings_with_text:
                    if h["level"] == "H1":
                        title = h["text"]
                        break
                if not title and headings_with_text:
                    title = headings_with_text[0]["text"]
                
                return {"title": title, "outline": headings_with_text}
                
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}
    
    def _extract_heading_content(self, heading: Dict, next_headings: List[Dict], page_lines: List[str]) -> str:
        """Extract content text that belongs to a specific heading."""
        heading_text = heading["text"]
        content_lines = []
        
        found_heading = False
        next_heading_found = False
        
        for line in page_lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is our current heading
            if heading_text in line and not found_heading:
                found_heading = True
                continue
            
            # If we found our heading, start collecting content
            if found_heading and not next_heading_found:
                # Check if this line is the next heading
                is_next_heading = False
                for other_heading in next_headings:
                    if other_heading["text"] in line:
                        is_next_heading = True
                        break
                
                if is_next_heading:
                    next_heading_found = True
                    break
                else:
                    # This line is content under our heading
                    if line != heading_text:  # Don't include the heading itself
                        content_lines.append(line)
        
        return '\n'.join(content_lines).strip()
    
    def analyze_pdfs(self, pdf_dir: str, persona: str, job: str, output_dir: str = None) -> Dict:
        """
        Analyze PDFs in a directory for similarity to a given persona and job.
        
        Args:
            pdf_dir: Directory containing PDF files
            persona: The persona for analysis
            job: The job description for analysis
            output_dir: Optional output directory for saving results
            
        Returns:
            dict: Analysis results
        """
        print(f"Analyzing PDFs in: {pdf_dir}")
        print(f"Persona: {persona}")
        print(f"Job: {job}")
        
        # Create query and embedding
        query = f"{persona}. {job}"
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Initialize metadata
        metadata = {
            "input_documents": [],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": str(datetime.now())
        }
        
        scored_sections = []
        all_outlines = {}  # Store outlines for later text extraction
        
        # Check if directory exists
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        # Process each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"\nProcessing: {pdf_file}")
            
            # Extract outline
            outline_data = self.extract_pdf_outline(pdf_path)
            all_outlines[pdf_file] = outline_data
            
            if not outline_data["title"] and not outline_data["outline"]:
                print(f"No content extracted from {pdf_file}")
                continue
            
            metadata["input_documents"].append(pdf_file)
            
            title = outline_data.get("title", "").strip()
            outline = outline_data.get("outline", [])
            
            print(f"Title: '{title}'")
            print(f"Outline items: {len(outline)}")
            
            # Process title
            if title:
                try:
                    score = util.cos_sim(query_embedding, self.model.encode(title, convert_to_tensor=True)).item()
                    scored_sections.append({
                        "document": pdf_file,
                        "page_number": 1,
                        "section_title": title,
                        "importance_rank": round(score, 4),
                        "ref_key": self.normalize(title),
                        "ref_type": "title"
                    })
                    print(f"Added title with score: {score:.4f}")
                except Exception as e:
                    print(f"Error processing title: {e}")
            
            # Process outline headings
            for item in outline:
                try:
                    heading = item["text"].strip()
                    page = item["page"]
                    
                    # Use heading + content for similarity scoring
                    content = item.get("content", "")
                    text_for_scoring = f"{heading} {content}".strip()
                    
                    score = util.cos_sim(query_embedding, self.model.encode(text_for_scoring, convert_to_tensor=True)).item()
                    scored_sections.append({
                        "document": pdf_file,
                        "page_number": page,
                        "section_title": heading,
                        "importance_rank": round(score, 4),
                        "ref_key": self.normalize(heading),
                        "ref_type": "heading",
                        "content": content
                    })
                    print(f"Added heading '{heading}' with score: {score:.4f}")
                except Exception as e:
                    print(f"Error processing outline item {item}: {e}")
        
        print(f"\nTotal scored sections: {len(scored_sections)}")
        
        # Sort by score and pick top K
        top_sections = sorted(scored_sections, key=lambda x: x["importance_rank"], reverse=True)[:self.top_k]
        print(f"Top {self.top_k} sections selected")
        
        # Extract refined text for top sections
        subsection_analysis = []
        
        for section in top_sections:
            pdf_file = section["document"]
            page_number = section["page_number"]
            heading_key = section["ref_key"]
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            print(f"\nExtracting text for: {section['section_title']} from {pdf_file}")
            
            try:
                # First try to get content from our extracted outline
                refined_text = ""
                if "content" in section and section["content"]:
                    refined_text = section["content"]
                else:
                    # Fallback to PyMuPDF extraction
                    doc = fitz.open(pdf_path)
                    page = doc[page_number - 1]
                    lines = page.get_text("text").split('\n')
                    
                    found = False
                    capture = False
                    buffer = []
                    
                    for line in lines:
                        norm_line = self.normalize(line)
                        
                        # Look for the heading in the normalized line
                        if not found and heading_key in norm_line:
                            found = True
                            capture = True
                            continue  # skip the heading line itself
                        
                        if capture:
                            # Check if we've hit another known heading
                            hit_next_heading = False
                            for s in scored_sections:
                                if (s["document"] == section["document"] and 
                                    s["page_number"] == page_number and 
                                    s["ref_key"] != heading_key and 
                                    s["ref_key"] in norm_line):
                                    hit_next_heading = True
                                    break
                            
                            if hit_next_heading:
                                break  # stop at next known heading
                                
                            if line.strip():  # Only add non-empty lines
                                buffer.append(line.strip())
                    
                    refined_text = " ".join(buffer).strip()
                    doc.close()
                
                if not refined_text:
                    refined_text = section["section_title"]
                
                subsection_analysis.append({
                    "document": section["document"],
                    "refined_text": refined_text,
                    "page_number": section["page_number"]
                })
                
                print(f"Successfully extracted text ({len(refined_text)} chars)")
                
            except Exception as e:
                print(f"Error extracting text from {pdf_file} page {page_number}: {e}")
                subsection_analysis.append({
                    "document": section["document"],
                    "refined_text": section["section_title"],
                    "page_number": section["page_number"]
                })
        
        # Prepare final output in the exact required format
        output = {
            "metadata": {
                "input_documents": metadata["input_documents"],
                "persona": metadata["persona"],
                "job_to_be_done": metadata["job_to_be_done"],
                "processing_timestamp": metadata["processing_timestamp"]
            },
            "extracted_sections": [
                {
                    "document": sec["document"],
                    "section_title": sec["section_title"],
                    "importance_rank": idx + 1,  # Rank starting from 1
                    "page_number": sec["page_number"]
                }
                for idx, sec in enumerate(top_sections)
            ],
            "subsection_analysis": [
                {
                    "document": sub["document"],
                    "refined_text": sub["refined_text"],
                    "page_number": sub["page_number"]
                }
                for sub in subsection_analysis
            ]
        }
        
        print(f"\nFinal output summary:")
        print(f"Input documents: {len(output['metadata']['input_documents'])}")
        print(f"Extracted sections: {len(output['extracted_sections'])}")
        print(f"Subsection analysis: {len(output['subsection_analysis'])}")
        
        # Save output if directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output.json")  # Changed filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)  # Changed to 4-space indent
            print(f"Output saved to {output_path}")
        
        return output


def load_challenge_config(config_file: str) -> Dict:
    """
    Load challenge configuration from JSON file.
    
    Args:
        config_file: Path to the JSON configuration file
        
    Returns:
        dict: Challenge configuration
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Loaded challenge config from {config_file}")
        return config
    except Exception as e:
        raise Exception(f"Failed to load challenge config from {config_file}: {e}")


def process_challenge_input(challenge_input: Dict, pdf_dir: str, output_dir: str = None, top_k: int = 5) -> Dict:
    """
    Process challenge input format and run PDF analysis.
    
    Args:
        challenge_input: Dictionary containing challenge info, persona, and job
        pdf_dir: Directory containing the PDF files
        output_dir: Optional output directory for saving results
        top_k: Number of top sections to extract
        
    Returns:
        dict: Analysis results
    """
    # Extract information from challenge input
    challenge_info = challenge_input.get("challenge_info", {})
    persona_info = challenge_input.get("persona", {})
    job_info = challenge_input.get("job_to_be_done", {})
    documents = challenge_input.get("documents", [])
    
    # Format persona and job
    persona = persona_info.get("role", "Travel Planner")
    job = job_info.get("task", "Plan travel itinerary")
    
    print(f"Challenge: {challenge_info.get('description', 'N/A')}")
    print(f"Test Case: {challenge_info.get('test_case_name', 'N/A')}")
    print(f"Documents to process: {[doc['filename'] for doc in documents]}")
    
    # Run analysis
    analyzer = IntegratedPDFAnalyzer(top_k=top_k)
    result = analyzer.analyze_pdfs(pdf_dir, persona, job, output_dir)
    
    # Add challenge info to result
    result["challenge_info"] = challenge_info
    result["expected_documents"] = [doc["filename"] for doc in documents]
    
    return result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDF Analyzer for similarity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python pdf_analyzer.py --pdf-dir ./pdfs --config ./config.json --output ./results

  # Direct analysis without config file
  python pdf_analyzer.py --pdf-dir ./pdfs --persona "Travel Planner" --job "Plan a 4-day trip" --output ./results

  # Specify number of top sections to extract
  python pdf_analyzer.py --pdf-dir ./pdfs --config ./config.json --output ./results --top-k 10
        """
    )
    
    parser.add_argument(
        '--pdf-dir', 
        required=True,
        help='Directory containing PDF files to analyze'
    )
    
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file containing challenge input'
    )
    
    parser.add_argument(
        '--persona',
        help='Persona for analysis (e.g., "Travel Planner"). Required if --config not provided.'
    )
    
    parser.add_argument(
        '--job',
        help='Job description for analysis. Required if --config not provided.'
    )
    
    parser.add_argument(
        '--output',
        help='Output directory for saving results (optional)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top sections to extract (default: 5)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=50,
        help='Maximum pages to process per PDF (default: 50)'
    )
    
    parser.add_argument(
        '--model-path',
        default="./models/all-MiniLM-L6-v2",
        help='Path to sentence transformer model (default: ./models/all-MiniLM-L6-v2)'
    )
    
    return parser.parse_args()


def main():
    """Main function with command line argument parsing."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.config and (not args.persona or not args.job):
        print("❌ Error: Either --config file must be provided, or both --persona and --job must be specified.")
        return 1
    
    if not os.path.exists(args.pdf_dir):
        print(f"❌ Error: PDF directory not found: {args.pdf_dir}")
        return 1
    
    try:
        if args.config:
            # Load configuration from file
            if not os.path.exists(args.config):
                print(f"❌ Error: Config file not found: {args.config}")
                return 1
            
            challenge_input = load_challenge_config(args.config)
            
            print("🚀 Running analysis with config file...")
            result = process_challenge_input(
                challenge_input=challenge_input,
                pdf_dir=args.pdf_dir,
                output_dir=args.output,
                top_k=args.top_k
            )
            
        else:
            # Direct analysis without config file
            print("🚀 Running direct analysis...")
            analyzer = IntegratedPDFAnalyzer(
                model_path=args.model_path,
                top_k=args.top_k,
                max_pages=args.max_pages
            )
            
            result = analyzer.analyze_pdfs(
                pdf_dir=args.pdf_dir,
                persona=args.persona,
                job=args.job,
                output_dir=args.output
            )
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Results summary:")
        print(f"   - Input documents: {len(result['metadata']['input_documents'])}")
        print(f"   - Top sections extracted: {len(result['extracted_sections'])}")
        print(f"   - Subsection analyses: {len(result['subsection_analysis'])}")
        
        if args.output:
            print(f"   - Results saved to: {os.path.join(args.output, 'output.json')}")
        
        # Print top sections for quick review
        print(f"\n🔍 Top {min(5, len(result['extracted_sections']))} most relevant sections:")
        for i, section in enumerate(result['extracted_sections'][:5], 1):
            print(f"   {i}. {section['document']} - {section['section_title']} (Rank: {section['importance_rank']})")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())