use clap::Parser;
use image::GenericImageView;
use linfa::Dataset;
use linfa::traits::Fit;
use linfa::prelude::Predict;
use linfa_clustering::KMeans;
use ndarray::{Array2, Array1, Axis};
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

    /// Output SVG swatch
    #[arg(short, long)]
    swatch: bool,

    /// SVG swatch output file (defaults to "swatch.svg" if not specified)
    #[arg(short, long, default_value = "swatch.svg")]
    output: String,
}

// New struct to hold color and its cluster size
#[derive(Debug)]
struct ColorInfo {
    rgb: [u8; 3],
    percentage: f64,
}

fn save_color_swatch(colours: &Vec<ColorInfo>, output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut svg = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 140">"#
    );

    for (i, color) in colours.iter().enumerate() {
        let x = i * 100;
        svg.push_str(&format!(
            r#"
    <rect x="{}" y="0" width="100" height="100" fill="rgb({}, {}, {})"/>
    <text x="{}" y="115" font-family="Arial" font-size="10" fill="black">{}, {}, {}</text>
    <text x="{}" y="130" font-family="Arial" font-size="10" fill="black">{:.1}%</text>"#,
            x, color.rgb[0], color.rgb[1], color.rgb[2],
            x + 5, color.rgb[0], color.rgb[1], color.rgb[2],
            x + 5, color.percentage
        ));
    }

    svg.push_str("\n</svg>");

    let mut file = File::create(output_file)?;
    file.write_all(svg.as_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("Loading image...");
    let img = image::open(&args.filename)?;
    
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
    )?;
    
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
    
    // Create vector of ColorInfo with percentages
    let mut colours: Vec<ColorInfo> = kmeans
        .centroids()
        .outer_iter()
        .enumerate()
        .map(|(i, cent)| ColorInfo {
            rgb: [cent[0] as u8, cent[1] as u8, cent[2] as u8],
            percentage: (cluster_sizes[i] as f64 / total_pixels) * 100.0,
        })
        .collect();
    
    // Sort by percentage (descending)
    colours.sort_by(|a, b| b.percentage.partial_cmp(&a.percentage).unwrap());

    // Print results
    println!("\nDominant colours (sorted by prevalence):");
    for color in &colours {
        println!(
            "RGB: ({}, {}, {}) - {:.1}% of image",
            color.rgb[0], color.rgb[1], color.rgb[2], color.percentage
        );
    }

    // Save swatch if requested
    if args.swatch {
        println!("\nSaving color swatch to {}...", args.output);
        save_color_swatch(&colours, &args.output)?;
    }
    
    Ok(())
}
