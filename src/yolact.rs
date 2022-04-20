//! For now this is a crude form of preprocessing and postprocessing.
//! I do not have time for the complete implementation right now.
//! Shoddy replacements:
//!  + Flood-fill: Replaces an id system I do not have time to pick apart.
//!  + Prototype mask: Don't have time to figure out the IOU mask algorithm.

use std::convert::TryInto;
use image::ImageBuffer;
use tokio::time::Instant;
use edgetpu::EdgeTpuContext;
use edgetpu::tflite::{self, InterpreterBuilder, FlatBufferModel, op_resolver::OpResolver, ops::builtin::BuiltinOpResolver, Interpreter};

pub struct Yolact<'a> {
    interpreter: Interpreter<'a, BuiltinOpResolver>
}
impl<'a> Yolact<'a> {
    pub fn init() -> Yolact<'a> {
        let model = FlatBufferModel::build_from_file(
            "data/FRC_model_edgetpu.tflite",
        ).expect("failed to load model");
    
        let resolver = BuiltinOpResolver::default();
        resolver.add_custom(edgetpu::custom_op(), edgetpu::register_custom_op());
    
        let builder = InterpreterBuilder::new(model, resolver).expect("must create interpreter builder");
    
        let edgetpu_context = EdgeTpuContext::open_device().expect("failed to open coral device");
    
        let mut interpreter = builder.build().expect("must build interpreter");
        interpreter.set_external_context(
            tflite::ExternalContextType::EdgeTpu,
            edgetpu_context.to_external_context(),
        );
        interpreter.set_num_threads(4); // Max
        interpreter.allocate_tensors().expect("failed to allocate tensors.");
        Yolact { interpreter }
    }

    pub(crate) fn classify(&mut self, frame_buffer: &mut [u32]) {
        classify(frame_buffer, &mut self.interpreter);
    }
}

struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>, //Bytes,
}

/// I don't have time to finish postprocessing so this will need to do for now
fn terrible_id(img: Vec<u8>) -> Vec<i8> {
    // Only run this on balls: For now it is the only item that needs unique ids
    let mut out = [-1; 28 * 28].to_vec();
    let mut id = -1;

    fn flood_fill(out: &mut Vec<i8>, start: usize, id: i8, img: &Vec<u8>) {
        let mut set = vec![start];
        
        while let Some(px) = set.pop() {
            if img.get(px - 1) == Some(&3) {
                out[px - 1] = id;
                set.push(px - 1);
            }
            if img.get(px + 1) == Some(&3) {
                out[px + 1] = id;
                set.push(px + 1);
            }
            if img.get(px - 28) == Some(&3) {
                out[px - 28] = id;
                set.push(px - 28);
            }
            if img.get(px + 28) == Some(&3) {
                out[px + 28] = id;
                set.push(px + 28);
            }
        }
    }

    for (px, class) in img.iter().cloned().enumerate() {
        if class == 3 && out[px] == -1 {
            id += 1;
            flood_fill(&mut out, px, id, &img);
        }
    }
    
    out
}

fn postprocess<'a>(results: Vec<Vec<f32>>) -> [u32; 224 * 224] {
    let dets = &results[4]; // first detection

    // I only am interested in the prototype masks for now 
    // (Not enough time or resources to complete the yolact detection cleanup implementation)
    // Right now this is essentially semantic segmentation

    /*
    // ident4.chunks(81).map(|chunk| chunk[0])
    // identity4.chunks(81).map(|chunk| chunk[1]) = Red Robot
    // identity4.chunks(81).map(|chunk| chunk[2]) = Blue Robot
    // identity4.chunks(81).map(|chunk| chunk[3]) = Ball

    println!("{:?}", dets.chunks(81).map(|chunk| chunk.iter().map(|a| *a).take(4) // Debugging
    //.filter(|a| *a >= 0.0)
    .collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>().as_slice().chunks(28).collect::<Vec<&[Vec<f32>]>>());
    */

    let classes: [u8; 28 * 28] = dets.chunks(81).map(|chunk| { // TODO differentiate between instances
        let mut max = 0.0;
        let cls: [bool; 4] = chunk.iter().take(4).map(|a| {*a > max && {max = *a; true}})
                .collect::<Vec<bool>>().as_slice().try_into().unwrap();
        match cls {
            [false, true, false, false] => 1, // Red robot
            [false, _, true, false] => 2,
            [false, _, _, true] => 3,
            [false, _, _, _] | [_, _, _, _] => 0, // TODO first element
        }
    }).collect::<Vec<u8>>().as_slice().try_into().unwrap();

    let ids = terrible_id(Vec::from(classes));

    /*for chunk in classes.as_slice().chunks(28).collect::<Vec<&[u8]>>().iter() {
        println!("{:?}", chunk);
    }*/ // Debugging model

    // TODO use confidence to shape mask // Note id: 0 is a ball id and the none id
    let sample: [u32; 224 * 224] = classes.iter().zip(&mut ids.into_iter()).map(|(cls, id)| [((*cls as u32) << 24 & (id as u32) << 16); 8]).flatten().collect::<Vec<u32>>().as_slice()
        .chunks(224).map(|chunk| [chunk; 8]).flatten().flatten().map(|r| *r).collect::<Vec<u32>>().try_into().unwrap();

    sample
}

fn classify_tile<'a>(frame_buffer: &mut [u32], interpreter: &mut Interpreter<'a, BuiltinOpResolver>) {
    let data = {
        frame_buffer.iter().map( |px| {
            let out: [u8; 3] = u32::to_be_bytes(*px)[..3].iter().map(|byte| *byte) //as f32 / 255.0)
                .collect::<Vec<u8>>().as_slice().try_into().unwrap();
            out
        }).flatten().collect::<Vec<u8>>()
    };
    // for now simply color
    let img = Image{
        width: 224, //640,
        height: 224, //480,
        channels: 3,
        data,
    };

    let tensor_index = interpreter.inputs()[0];
    let required_shape = interpreter.tensor_info(tensor_index).unwrap().dims;
    if img.height != required_shape[1]
            || img.width != required_shape[2]
            || img.channels != required_shape[3] {
        eprintln!("Input size mismatches:");
        eprintln!("\twidth: {} vs {}", img.width, required_shape[0]);
        eprintln!("\theight: {} vs {}", img.height, required_shape[1]);
        eprintln!("\tchannels: {} vs {}", img.channels, required_shape[2]);
    }

    let _start_time = Instant::now();
    interpreter.tensor_data_mut(tensor_index).unwrap()
        .copy_from_slice(img.data.as_ref());
    interpreter.invoke().expect("invoke failed");
    println!("eval time: {}μs", _start_time.elapsed().as_micros()); // ~50ms on edgetpu

    let outputs = interpreter.outputs();
    let mut results: Vec<Vec<f32>> = Vec::new();

    for &output in outputs {
       let tensor_info = interpreter.tensor_info(output).expect("must data");
       match tensor_info.element_kind {
           tflite::context::ElementKind::kTfLiteUInt8 => { // TODO
                let out_tensor: &[u8] = interpreter.tensor_data(output).expect("must data");
                let scale = tensor_info.params.scale;
                let zero_point = tensor_info.params.zero_point;
                results.push(out_tensor.into_iter()
                    .map(|&x| scale * (((x as i32) - zero_point) as f32)).collect());
            }
            tflite::context::ElementKind::kTfLiteFloat32 => {
                let out_tensor: &[f32] = interpreter.tensor_data(output).expect("must data");
                results.push(out_tensor.into_iter().copied().collect());
            }
            _ => eprintln!(
                "Tensor {} has unsupported output type {:?}.",
                tensor_info.name, tensor_info.element_kind,
            ),
        }
    }
    frame_buffer.copy_from_slice(&postprocess(results));
}

pub(crate) fn classify<'a>(frame_buffer: &mut [u32], interpreter: &mut Interpreter<'a, BuiltinOpResolver>) {
    // println!("Classifying");

    let data = {
        frame_buffer.iter().map( |px| {
            let out: [u8; 3] = u32::to_be_bytes(*px)[..3].iter().map(|byte| *byte) //as f32 / 255.0)
                .collect::<Vec<u8>>().as_slice().try_into().unwrap();
            out
        }).flatten().collect::<Vec<u8>>()
    };

    // // Debugging model
    // let image = image::DynamicImage::ImageRgb8(image::io::Reader::open("data/red_robot.png").unwrap().decode().unwrap().to_rgb8());

    // Using a library to scale images.
    let image = image::DynamicImage::ImageRgb8(ImageBuffer::from_vec(640, 480, data).unwrap());
    let mut new_image = image.resize_exact(448, 224, image::imageops::FilterType::Triangle);

    // TODO: tile the image
    // 640 * 480 image
    // 224 * 224 sample
    let mut t1 = new_image.clone().crop(0, 0, 224, 224).as_bytes().chunks(3).map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0])).collect::<Vec<u32>>();
    let mut t2 = new_image.crop(224, 0, 224, 224).as_bytes().chunks(3).map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0])).collect::<Vec<u32>>();

    classify_tile(&mut t1, interpreter);
    classify_tile(&mut t2, interpreter);

    let img_buf = t1.chunks(224).collect::<Vec<&[u32]>>().iter().zip(t2.chunks(224).collect::<Vec<&[u32]>>().iter())
        .map(|(t1_horiz, t2_horiz)| [*t1_horiz, *t2_horiz].concat()).flatten(/*check op*/).collect::<Vec<u32>>();
    
    let data = {
        img_buf.iter().map( |px| {
            let out: [u8; 3] = u32::to_be_bytes(*px)[..3].iter().map(|byte| *byte) //as f32 / 255.0)
                .collect::<Vec<u8>>().as_slice().try_into().unwrap();
            out
        }).flatten().collect::<Vec<u8>>()
    };

    let image = image::DynamicImage::ImageRgb8(ImageBuffer::from_vec(448, 224, data).unwrap());
    let class_be = image.resize_exact(640, 480, image::imageops::FilterType::Triangle).as_bytes()
        .chunks(3).map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0])).collect::<Vec<u32>>();
    frame_buffer.copy_from_slice(&class_be);
}