use anyhow::{Result, Context};
use clap::Parser;
use clap::ValueEnum;
use image::GenericImageView;
use linfa::Dataset;
use linfa::prelude::Predict;
use linfa::traits::Fit;
use linfa_clustering::KMeans;
use ndarray::{Array2, Array1, Axis};
use serde_json;
use std::fs::File;
use std::io::Write;

#[derive(Parser)]
#[command(name = "dominant-colours")]
#[command(about = "Extract dominant colours from images using k-means clustering")]
struct Args {
    /// Image file to analyze
    filename: String,

    /// Number of colours to extract
    #[arg(short, long, default_value_t = 6)]
    colours: usize,

    /// Output format for colour data
    #[arg(short = 'f', long, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    /// Output SVG swatch
    #[arg(short, long)]
    swatch: bool,

    /// SVG swatch output file (defaults to "swatch.svg" if not specified)
    #[arg(short, long, default_value = "swatch.svg")]
    output: String,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum OutputFormat {
    Text,
    Json,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Text => write!(f, "text"),
            OutputFormat::Json => write!(f, "json"),
        }
    }
}

// Struct to hold colour and its cluster size
#[derive(Debug)]
struct ColourInfo {
    rgb: [u8; 3],
    percentage: f64,
}

fn rgb_to_hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

fn save_colour_swatch(colours: &Vec<ColourInfo>, output_file: &str) -> Result<()> {
    let mut svg = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 140">"#
    );

    for (i, colour) in colours.iter().enumerate() {
        let x = i * 100;
        svg.push_str(&format!(
            r#"
    <rect x="{}" y="0" width="100" height="100" fill="rgb({}, {}, {})"/>
    <text x="{}" y="115" font-family="Arial" font-size="10" fill="black">{}, {}, {}</text>
    <text x="{}" y="130" font-family="Arial" font-size="10" fill="black">{:.1}%</text>"#,
            x, colour.rgb[0], colour.rgb[1], colour.rgb[2],
            x + 5, colour.rgb[0], colour.rgb[1], colour.rgb[2],
            x + 5, colour.percentage
        ));
    }

    svg.push_str("\n</svg>");

    let mut file = File::create(output_file)?;
    file.write_all(svg.as_bytes())?;
    Ok(())
}

fn analyze_image(args: &Args) -> Result<Vec<ColourInfo>> {
    println!("Loading image...");
    let img = image::open(&args.filename)
        .context("Failed to open image file")?;

    println!("Resizing image...");
    let resized = img.resize(150, 150, image::imageops::FilterType::Lanczos3);

    println!("Converting to pixels...");
    let pixels: Vec<[f64; 3]> = resized.pixels()
        .map(|(_, _, rgb)| [
            rgb[0] as f64,
            rgb[1] as f64,
            rgb[2] as f64,
        ])
        .collect();

    println!("Preparing data for clustering...");
    let data = Array2::from_shape_vec(
        (pixels.len(), 3),
        pixels.into_iter().flatten().collect(),
    ).context("Failed to create data array")?;

    let targets: Array1<f64> = Array1::zeros(data.len_of(Axis(0)));
    let dataset = Dataset::new(data.clone(), targets);

    println!("Running k-means clustering...");
    let kmeans = KMeans::params(args.colours)
        .max_n_iterations(100)
        .fit(&dataset)?;

    println!("Analyzing clusters...");
    // Get cluster assignments for each pixel
    let predictions = kmeans.predict(&dataset);
    let total_pixels = predictions.len() as f64;

    // Count pixels in each cluster
    let mut cluster_sizes = vec![0; args.colours];
    for &cluster in predictions.iter() {
        cluster_sizes[cluster] += 1;
    }

    // Create vector of ColourInfo with percentages
    let mut colours: Vec<ColourInfo> = kmeans
        .centroids()
        .outer_iter()
        .enumerate()
        .map(|(i, cent)| ColourInfo {
            rgb: [cent[0] as u8, cent[1] as u8, cent[2] as u8],
            percentage: (cluster_sizes[i] as f64 / total_pixels) * 100.0,
        })
        .collect();

    // Sort by percentage (descending)
    colours.sort_by(|a, b| b.percentage.partial_cmp(&a.percentage).unwrap());

    Ok(colours)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let colours = analyze_image(&args)?;

    match args.format {
        OutputFormat::Text => {
            println!("\nDominant colours (sorted by prevalence):");
            for colour in &colours {
                println!(
                    "RGB: ({}, {}, {}) - {:.1}% of image",
                    colour.rgb[0], colour.rgb[1], colour.rgb[2], colour.percentage
                );
            }
        }
        OutputFormat::Json => {
            let json = serde_json::json!({
                "colours": colours.iter().map(|c| {
                    serde_json::json!({
                        "rgb": c.rgb,
                        "percentage": c.percentage,
                        "hex": rgb_to_hex(c.rgb)
                    })
                }).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
    }

    // Save swatch if requested
    if args.swatch {
        println!("\nSaving colour swatch to {}...", args.output);
        save_colour_swatch(&colours, &args.output)
            .context("Failed to save colour swatch")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::*;
    use image::{RgbImage, Rgb};
    use tempfile::tempdir;

    #[test]
    fn test_arg_parsing() {
        let args = Args::parse_from(["program", "test.jpg"]);
        assert_eq!(args.filename, "test.jpg");
        assert_eq!(args.colours, 6); // default value
        assert!(!args.swatch); // default false

        let args = Args::parse_from(["program", "-c", "8", "test.jpg"]);
        assert_eq!(args.colours, 8);

        let args = Args::parse_from(["program", "--swatch", "-o", "custom.svg", "test.jpg"]);
        assert!(args.swatch);
        assert_eq!(args.output, "custom.svg");
    }

#[test]
fn test_format_arg_parsing() {
    let args = Args::parse_from(["program", "test.jpg"]);
    assert_eq!(args.format, OutputFormat::Text);  // default value

    let args = Args::parse_from(["program", "-f", "json", "test.jpg"]);
    assert_eq!(args.format, OutputFormat::Json);

    let args = Args::parse_from(["program", "--format", "json", "test.jpg"]);
    assert_eq!(args.format, OutputFormat::Json);
}

#[test]
fn test_json_output() -> Result<(), Box<dyn std::error::Error>> {
    // Create a small test image like in test_dominant_colour_extraction
    let img = RgbImage::new(100, 100);
    // Fill with known colours...

    // Save temporary image
    let temp_dir = tempdir()?;
    let image_path = temp_dir.path().join("test_image.png");
    img.save(&image_path)?;

    // Run analysis with JSON output
    let args = Args {
        filename: image_path.to_str().unwrap().to_string(),
        colours: 3,
        format: OutputFormat::Json,
        swatch: false,
        output: "".to_string(),
    };

    let colours = analyze_image(&args)?;

    // Convert to JSON and verify structure
    let json = serde_json::json!({
        "colours": colours.iter().map(|c| {
            serde_json::json!({
                "rgb": c.rgb,
                "percentage": c.percentage,
                "hex": rgb_to_hex(c.rgb)
            })
        }).collect::<Vec<_>>()
    });

    let json_str = serde_json::to_string_pretty(&json)?;

    // Verify JSON structure
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    assert!(parsed.is_object());
    assert!(parsed.get("colours").unwrap().is_array());

    // Check first colour has required fields
    let first_colour = &parsed["colours"][0];
    assert!(first_colour.get("rgb").is_some());
    assert!(first_colour.get("percentage").is_some());
    assert!(first_colour.get("hex").is_some());

    Ok(())
}

#[test]
fn test_rgb_to_hex() {
    assert_eq!(rgb_to_hex([255, 255, 255]), "#ffffff");
    assert_eq!(rgb_to_hex([0, 0, 0]), "#000000");
    assert_eq!(rgb_to_hex([255, 0, 0]), "#ff0000");
    assert_eq!(rgb_to_hex([0, 255, 0]), "#00ff00");
    assert_eq!(rgb_to_hex([0, 0, 255]), "#0000ff");
    assert_eq!(rgb_to_hex([85, 85, 85]), "#555555");
}

    #[test]
    fn test_colour_swatch_generation() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = tempdir()?;
        let output_path = temp_dir.path().join("test_swatch.svg");

        let colours = vec![
            ColourInfo {
                rgb: [255, 0, 0],
                percentage: 50.0,
            },
            ColourInfo {
                rgb: [0, 255, 0],
                percentage: 30.0,
            },
            ColourInfo {
                rgb: [0, 0, 255],
                percentage: 20.0,
            },
        ];

        save_colour_swatch(&colours, output_path.to_str().unwrap())?;

        // Verify file exists and contains expected content
        let content = std::fs::read_to_string(output_path)?;
        assert!(content.contains("rgb(255, 0, 0)"));
        assert!(content.contains("rgb(0, 255, 0)"));
        assert!(content.contains("rgb(0, 0, 255)"));
        assert!(content.contains("50.0%"));
        assert!(content.contains("30.0%"));
        assert!(content.contains("20.0%"));

        Ok(())
    }

    #[test]
    fn test_dominant_colour_extraction() -> Result<(), Box<dyn std::error::Error>> {
        // Create a test image with known colours
        let mut img = RgbImage::new(100, 100);

        // Fill image with 50% red, 30% green, 20% blue
        for y in 0..100 {
            for x in 0..100 {
                let pixel = if y < 50 {
                    Rgb([255, 0, 0])      // Red (50%)
                } else if y < 80 {
                    Rgb([0, 255, 0])      // Green (30%)
                } else {
                    Rgb([0, 0, 255])      // Blue (20%)
                };
                img.put_pixel(x, y, pixel);
            }
        }

        // Save temporary image
        let temp_dir = tempdir()?;
        let image_path = temp_dir.path().join("test_image.png");
        img.save(&image_path)?;

        // Run analysis
        let args = Args {
            filename: image_path.to_str().unwrap().to_string(),
            colours: 3,
            format: OutputFormat::Text,
            swatch: false,
            output: "".to_string(),
        };

        let colours = analyze_image(&args)?;

        // Verify results (with some tolerance for k-means variation)
        assert!(colours.len() == 3);

        // Sort colours by percentage for stable comparison
        let mut sorted_colours = colours;
        sorted_colours.sort_by(|a, b| b.percentage.partial_cmp(&a.percentage).unwrap());

        // Check percentages (with tolerance)
        assert!((sorted_colours[0].percentage - 50.0).abs() < 5.0);
        assert!((sorted_colours[1].percentage - 30.0).abs() < 5.0);
        assert!((sorted_colours[2].percentage - 20.0).abs() < 5.0);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        // Test non-existent file
        let result = image::open("non_existent.jpg");
        assert!(result.is_err());

        // Test invalid colour count
        let args = Args {
            filename: "test.jpg".to_string(),
            colours: 0,  // Invalid number of colours
            format: OutputFormat::Text,
            swatch: false,
            output: "".to_string(),
        };

        let result = analyze_image(&args);
        assert!(result.is_err());
    }
}
