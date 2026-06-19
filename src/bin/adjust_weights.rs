use anyhow::Result;
use clap::Parser;
use shogi_ai::evaluation::SparseModel;
use std::path::PathBuf;

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
    material: Option<f32>,

    /// Scaling factor for all weights (w and bias)
    #[arg(long, default_value_t = 1.0)]
    w_scale: f32,

    /// Optional second weight file. Output becomes input + blend_ratio * (blend_target - input).
    #[arg(long)]
    blend_target: Option<PathBuf>,

    /// Blend ratio used with --blend-target.
    #[arg(long, default_value_t = 1.0)]
    blend_ratio: f32,
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

    if let Some(blend_target) = &args.blend_target {
        let mut target = SparseModel::new(0.0, 0.0);
        println!("Loading blend target from: {:?}", blend_target);
        target.load(blend_target)?;
        model.bias += args.blend_ratio * (target.bias - model.bias);
        model.material_coeff += args.blend_ratio * (target.material_coeff - model.material_coeff);
        for (weight, target_weight) in model.w.iter_mut().zip(target.w.iter()) {
            *weight += args.blend_ratio * (*target_weight - *weight);
        }
        println!("Blend ratio: {}", args.blend_ratio);
    }

    // Adjust the parameters
    if let Some(material) = args.material {
        model.material_coeff = material;
    }
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
