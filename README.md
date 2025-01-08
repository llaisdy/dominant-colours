# dominant-colours

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command line tool that extracts dominant colours from images using k-means clustering. Unlike simpler color quantization algorithms, this tool uses machine learning to find natural color clusters and reports their prevalence in the image.

## Features

- Uses k-means clustering to find natural color groupings
- Reports the percentage of the image occupied by each color
- Generates SVG color swatches
- Fast execution using optimized Rust libraries

## Installation

```bash
cargo install --path .
```

## Usage

```bash
# Basic usage - prints RGB values
dominant-colours image.jpg

# Specify number of colors to extract
dominant-colours -c 8 image.jpg

# Generate a color swatch
dominant-colours --swatch image.jpg

# Specify swatch output file
dominant-colours -s -o my-colors.svg image.jpg
```

## Dependencies

- image: Image loading and manipulation
- linfa-clustering: Machine learning algorithms
- clap: Command line argument parsing
