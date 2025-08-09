
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use shogi_ai::evaluation::SparseModel;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input weight file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output weight file path
    #[arg(short, long)]
    output: PathBuf,

    /// New value for material_coeff
    #[arg(long)]
    material: f32,

    /// Scaling factor for all weights (w and bias)
    #[arg(long)]
    w_scale: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load the model from the input file
    let mut model = SparseModel::new(0.0, 0.0); // eta and lambda are not used here
    println!("Loading weights from: {:?}", args.input);
    model.load(&args.input)?;
    println!("Weights loaded successfully.");
    println!("Original material_coeff: {}", model.material_coeff);
    println!("Original bias: {}", model.bias);


    // Adjust the parameters
    model.material_coeff = args.material;
    model.bias *= args.w_scale;
    for w in model.w.iter_mut() {
        *w *= args.w_scale;
    }
    println!("Parameters adjusted.");
    println!("New material_coeff: {}", model.material_coeff);
    println!("New bias: {}", model.bias);


    // Save the adjusted model to the output file
    println!("Saving adjusted weights to: {:?}", args.output);
    model.save(&args.output)?;
    println!("Adjusted weights saved successfully.");

    Ok(())
}
