# PDF Analyzer 🔍

A powerful PDF analysis tool that performs similarity analysis on PDF documents using sentence transformers. This tool is designed to extract and analyze the most relevant sections from PDF files based on specific personas and job descriptions.

## Features

- **Similarity Analysis**: Uses sentence transformer models to find relevant content
- **Flexible Input**: Support for both configuration files and direct parameters
- **Persona-Based Analysis**: Analyze PDFs from different perspectives (Travel Planner, etc.)
- **Configurable Output**: Control the number of top sections and pages to process
- **Docker Support**: Containerized for easy deployment and consistency

## Prerequisites

- Docker (for containerized usage)
- Python 3.x (for direct usage)
- Sentence transformer models

## Installation

### Using Docker (Recommended)

```bash
# Pull or build the Docker image
docker build -t pdfanalyzer:3.0 .
```

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd pdf-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Docker Usage

#### Basic usage with config file
```bash
docker run pdfanalyzer:3.0 \
  --pdf-dir ./pdfs \
  --config ./config.json \
  --output ./results
```

#### Direct analysis without config file
```bash
docker run pdfanalyzer:3.0 \
  --pdf-dir ./pdfs \
  --persona "Travel Planner" \
  --job "Plan a 4-day trip" \
  --output ./results
```

#### Advanced usage with custom parameters
```bash
docker run pdfanalyzer:3.0 \
  --pdf-dir ./pdfs \
  --config ./config.json \
  --output ./results \
  --top-k 10 \
  --max-pages 100 \
  --model-path ./custom-models/sentence-transformer
```

### Local Usage

```bash
# Basic usage with config file
python 1B.py --pdf-dir ./pdfs --config ./config.json --output ./results

# Direct analysis without config file
python 1B.py --pdf-dir ./pdfs --persona "Travel Planner" --job "Plan a 4-day trip" --output ./results

# Specify number of top sections to extract
python 1B.py --pdf-dir ./pdfs --config ./config.json --output ./results --top-k 10
```

## Configuration

### Command Line Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--pdf-dir` | Directory containing PDF files to analyze | Yes | - |
| `--config` | Path to JSON configuration file | No* | - |
| `--persona` | Persona for analysis (e.g., "Travel Planner") | No* | - |
| `--job` | Job description for analysis | No* | - |
| `--output` | Output directory for saving results | No | - |
| `--top-k` | Number of top sections to extract | No | 5 |
| `--max-pages` | Maximum pages to process per PDF | No | 50 |
| `--model-path` | Path to sentence transformer model | No | ./models/all-MiniLM-L6-v2 |

*Either `--config` must be provided, OR both `--persona` and `--job` must be specified.

### Configuration File Format

Create a JSON configuration file with the following structure:

```json
{
  "persona": "Travel Planner",
  "job": "Plan a comprehensive 4-day trip to Tokyo including accommodations, attractions, and dining recommendations",
  "additional_parameters": {
    "focus_areas": ["hotels", "restaurants", "attractions"],
    "budget_range": "mid-range"
  }
}
```

## Examples

### Travel Planning Analysis
```bash
docker run pdfanalyzer:3.0 \
  --pdf-dir ./travel-guides \
  --persona "Travel Planner" \
  --job "Create a detailed itinerary for a 5-day cultural trip to Paris" \
  --output ./travel-results \
  --top-k 8
```

### Research Document Analysis
```bash
docker run pdfanalyzer:3.0 \
  --pdf-dir ./research-papers \
  --persona "Research Analyst" \
  --job "Identify key findings and methodologies in machine learning papers" \
  --output ./research-analysis \
  --max-pages 30
```

### Business Document Review
```bash
docker run pdfanalyzer:3.0 \
  --pdf-dir ./business-docs \
  --config ./business-config.json \
  --output ./business-analysis \
  --top-k 15
```

## Output

The tool generates analysis results in the specified output directory, including:

- Extracted relevant sections from PDFs
- Similarity scores and rankings
- Summary reports based on the specified persona and job requirements

## Model Information

The default sentence transformer model used is `all-MiniLM-L6-v2`, which provides a good balance between performance and accuracy. You can specify a custom model path using the `--model-path` parameter.

## Performance Considerations

- **PDF Size**: Large PDFs are limited by the `--max-pages` parameter (default: 50 pages)
- **Top-K Selection**: Adjust `--top-k` based on your needs (default: 5 sections)
- **Model Selection**: Different sentence transformer models offer varying trade-offs between speed and accuracy

## Troubleshooting

### Common Issues

1. **PDF Directory Not Found**: Ensure the PDF directory path is correct and accessible
2. **Model Loading Error**: Verify the sentence transformer model path exists
3. **Memory Issues**: Reduce `--max-pages` or `--top-k` for large document sets
4. **Empty Results**: Check that PDFs contain extractable text content

### Docker Issues

```bash
# Check if container is running properly
docker logs <container_id>

# Verify volume mounts for file access
docker run -v /host/path:/container/path pdfanalyzer:3.0 --help
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the example configurations
