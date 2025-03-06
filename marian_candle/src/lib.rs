use anyhow::Error as E;
use candle_core::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::marian;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokenizers::Tokenizer;

#[no_mangle]
pub extern "C" fn translate_text(input: *const c_char, output: *mut c_char, max_length: usize) {
    let input_str = unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("");
    
    if input_str.is_empty() {
        eprintln!("Invalid input.");
        return;
    }

    let args = Args {
        model: None,
        tokenizer: None,
        tokenizer_dec: None,
        which: Which::Base,
        cpu: false,
        quantized: false,
        text: input_str.to_string(),
    };

    match translate(args) {
        Ok(translation) => {
            let c_str = CString::new(translation).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(c_str.as_ptr(), output, max_length);
            }
        }
        Err(e) => {
            eprintln!("Translation error: {:?}", e);
            let error_message = format!("Error: {:?}", e);
            let c_str = CString::new(error_message).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(c_str.as_ptr(), output, max_length);
            }
        }
    }
}

#[derive(Clone, Debug, Copy, clap::ValueEnum)]
enum Which {
    Base,
    Big,
}

#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long)]
    tokenizer_dec: Option<String>,
    #[arg(long, default_value = "base")]
    which: Which,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    quantized: bool,
    #[arg(long)]
    text: String,
}

fn translate(args: Args) -> anyhow::Result<String> {
    use hf_hub::api::sync::Api;
    let config = match args.which {
        Which::Base => marian::Config::opus_mt_fr_en(),
        Which::Big => marian::Config::opus_mt_tc_big_fr_en(),
    };
    let tokenizer = {
        let tokenizer = match args.tokenizer {
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
            None => {
                let name = match args.which {
                    Which::Base => "tokenizer-marian-base-fr.json",
                    Which::Big => "tokenizer-marian-fr.json",
                };
                Api::new()?
                    .model("lmz/candle-marian".to_string())
                    .get(name)?
            }
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };
    let tokenizer_dec = {
        let tokenizer = match args.tokenizer_dec {
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
            None => {
                let name = match args.which {
                    Which::Base => "tokenizer-marian-base-en.json",
                    Which::Big => "tokenizer-marian-en.json",
                };
                Api::new()?
                    .model("lmz/candle-marian".to_string())
                    .get(name)?
            }
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };
    let mut tokenizer_dec = TokenOutputStream::new(tokenizer_dec);
    let device = candle_examples::device(args.cpu)?;
    let vb = {
        let model = match args.model {
            Some(model) => std::path::PathBuf::from(model),
            None => match args.which {
                Which::Base => Api::new()?
                    .repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-fr-en".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/4".to_string(),
                    ))
                    .get("model.safetensors")?,
                Which::Big => Api::new()?
                    .model("Helsinki-NLP/opus-mt-tc-big-fr-en".to_string())
                    .get("model.safetensors")?,
            },
        };
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model], DType::F32, &device)? }
    };
    let mut model = marian::MTModel::new(&config, vb)?;
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);
    let encoder_xs = {
        let mut tokens = tokenizer
            .encode(args.text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        tokens.push(config.eos_token_id);
        let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        model.encoder().forward(&tokens, 0)?
    };
    let mut token_ids = vec![config.decoder_start_token_id];
    for index in 0..1000 {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.decode(&input_ids, &encoder_xs, start_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        token_ids.push(token);
        if let Some(t) = tokenizer_dec.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if token == config.eos_token_id || token == config.forced_eos_token_id {
            break;
        }
    }
    if let Some(rest) = tokenizer_dec.decode_rest().map_err(E::msg)? {
        print!("{rest}");
        Ok(rest)
    } else {
        Ok(String::new())
    }
}