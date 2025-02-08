use plotters::prelude::*;
use rand::Rng;
use std::env::current_exe;
use std::fs::File;
use std::io::Read;
use std::vec;

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,

    last_input: Vec<f64>,
    last_output: Vec<f64>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();

        let weights = (0..output_size) // For each node in the current layer
            .map(|_| {
                (0..input_size) // And for each input to that node
                    .map(|_| rng.random_range(-1.0..1.0)) // Select a random number in our seed range
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        let biases = (0..output_size) // For each node
            .map(|_| rng.random_range(-1.0..1.0)) // Select a random number in our seed range
            .collect::<Vec<f64>>();

        Self {
            weights,
            biases,
            last_input: vec![],
            last_output: vec![],
        }
    }

    // https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    fn sigmoid_derivative(sigmoid_x: f64) -> f64 {
        sigmoid_x * (1.0 - sigmoid_x)
    }

    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.last_input = input.clone();
        let mut output = vec![];

        for (weights, bias) in self.weights.iter().zip(self.biases.iter()) {
            // z = sum(w * x) + bias for w,x in weights,input
            let z_i: f64 = weights
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + bias;

            let a_i = Self::sigmoid(z_i);
            output.push(a_i);
        }
        self.last_output = output.clone();
        output
    }

    fn backward(&mut self, dl_dout: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        // dL/dz = dL/dout * sigmoid_derivative(z)
        let dl_dz: Vec<f64> = dl_dout
            .iter()
            .zip(self.last_output.iter())
            .map(|(&d, &a)| d * Self::sigmoid_derivative(a))
            .collect();

        // dL/db (Gradient for the bias) = dL/dz
        let dl_db = dl_dz.to_vec();

        // dL/dw (Gradient for the weight) = dL/dz * x for
        let input = &self.last_input;
        let mut dl_dw = vec![vec![0.0; input.len()]; self.weights.len()];
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                dl_dw[i][j] = dl_dz[i] * input[j];
            }
        }

        // Update weights and biases using gradient descent
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                self.weights[i][j] -= learning_rate * dl_dw[i][j];
            }
            self.biases[i] -= learning_rate * dl_db[i];
        }

        // Compute and return dL/dx for previous layer
        // dL/dx = sum(dL/dz * w) for w in weights[i]
        let mut dl_dinput = vec![0.0; input.len()];
        for j in 0..input.len() {
            let mut sum = 0.0;
            for i in 0..self.weights.len() {
                sum += dl_dz[i] * self.weights[i][j];
            }
            dl_dinput[j] = sum;
        }
        dl_dinput
    }
}

struct Network {
    layers: Vec<Layer>,
}

impl Network {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = vec![];
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { layers }
    }

    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output)
        }
        output
    }

    fn backward(&mut self, dl_dout: &Vec<f64>, learning_rate: f64) {
        let mut grad = dl_dout.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }
    }
}

fn normalize_images(input: Vec<Vec<Vec<u8>>>) -> Vec<Vec<Vec<f64>>> {
    input
        .into_iter()
        .map(|vec1| {
            vec1.into_iter()
                .map(|vec2| vec2.into_iter().map(|item| item as f64 / 255.0).collect())
                .collect()
        })
        .collect()
}

fn flatten_images(input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
    input
        .into_iter()
        .map(|vec1| vec1.into_iter().flatten().collect())
        .collect()
}

fn flatten_labels(input: Vec<u8>) -> Vec<Vec<f64>> {
    let mut output = vec![];
    for item in input {
        let mut current_vec = vec![0.0; 10];
        current_vec[item as usize] = 1.0;
        output.push(current_vec);
    }
    output
}

// https://github.com/R34ll/mnist_rust
fn load_dataset_mnist_images(path: &str) -> Vec<Vec<Vec<f64>>> {
    let mut file = File::open(path).expect("File non find");

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    data.drain(0..16); // https://github.com/busyboredom/rust-mnist/blob/main/src/lib.rs#L185
    let dataset: Vec<Vec<Vec<u8>>> = data
        .chunks(784)
        .map(|chunk| chunk.chunks(28).map(|s| s.into()).collect())
        .collect();
    println!("Images in dataset: {}", dataset.len());
    normalize_images(dataset)
}

// https://github.com/busyboredom/rust-mnist/blob/main/src/lib.rs#L185
fn load_dataset_mnist_label(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect("File non find");

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    data.drain(0..8);
    let labels: Vec<u8> = data.iter_mut().map(|&mut l| l as u8).collect();
    println!("Labels in dataset: {}", labels.len());
    labels
}

fn draw_mnist(image: Vec<Vec<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("plot.png", (840, 840)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let (width, height) = (image[0].len(), image.len());

    // Define the grid layout
    let cell_width: i32 = 840 / width as i32;
    let cell_height: i32 = 840 / height as i32;

    for (i, row) in image.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            let color = HSLColor(0.0, 0.0, value);

            root_area.draw(&Rectangle::new(
                [
                    (j as i32 * cell_width, i as i32 * cell_height),
                    ((j + 1) as i32 * cell_width, (i + 1) as i32 * cell_height),
                ],
                color.filled(),
            ))?;
        }
    }

    root_area.present()?;
    Ok(())
}

fn draw_loss_over_time(losses: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Create a drawing area for the plot
    let root = BitMapBackend::new("line_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // Define the chart area and label axes
    let mut chart = ChartBuilder::on(&root)
        .caption("Loss over epochs", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..losses.len() as f64, 0.0..0.2)?;
    // .build_cartesian_2d(0.0..losses.len() as f64, (0.0..0.2).log_scale())?;

    // Draw the mesh for the chart
    chart.configure_mesh().draw()?; // Define the data points for the line

    // Plot the line using the data points
    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(i, &loss)| (i as f64, loss)),
            &RED,
        ))?
        .label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Add legend to the chart
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Save the plot
    root.present()?;
    Ok(())
}

fn rmse(current: &Vec<f64>, target: &Vec<f64>) -> f64 {
    let n = current.len();
    let sum_squared_error: f64 = current
        .iter()
        .zip(target.iter())
        .map(|(c, t)| (c - t).powi(2))
        .sum();
    (sum_squared_error / n as f64).sqrt()
}

fn main() {
    let mut losses: Vec<f64> = vec![];

    //#########################################################################
    // Load data
    //#########################################################################
    let dataset: Vec<Vec<Vec<f64>>> =
        load_dataset_mnist_images("datasets/mnist/t10k-images.idx3-ubyte");
    let flat_dataset = flatten_images(dataset.clone());
    let labels: Vec<u8> = load_dataset_mnist_label("datasets/mnist/t10k-labels.idx1-ubyte");
    let flat_labels = flatten_labels(labels.clone());

    //#########################################################################
    // Create network
    //#########################################################################

    let mut network = Network::new(&[784, 50, 10]);
    let learning_rate = 0.5;
    let epochs = 100;

    //#########################################################################
    // Training
    //#########################################################################

    for epoch in 0..epochs {
        let mut loss_sum = 0.0;
        for i in 0..flat_dataset.len() {
            let output = network.forward(&flat_dataset[i]);
            let loss = rmse(&output, &flat_labels[i]); // mean squared error
            loss_sum += loss;

            // dl/dout (Gradient of loss for the network) = (output - target)
            let dl_dout = output.into_iter().zip(flat_labels[i].clone().into_iter()).map(|(o,t)| (o-t)).collect();
            network.backward(&dl_dout, learning_rate);
        }
        let loss_avg = loss_sum / flat_dataset.len() as f64;
        losses.push(loss_avg);
        if epoch % 10 == 0 {
            println!("Epoch {}: avg loss = {}", epoch, loss_avg);
        }
    }

    //#########################################################################
    // Testing
    //#########################################################################

    // for (input, target) in training_data.iter() {
    //     let output = network.forward(input);
    //     println!(
    //         "Input: {:?}, target: {:?} output: {:?}",
    //         input, target, output
    //     );
    // }

    //#########################################################################
    // Visualization
    //#########################################################################
    let _ = draw_loss_over_time(losses);

    //  MNIST
    let data: Vec<Vec<f64>> = dataset[0].clone();
    let _ = draw_mnist(data);
}

// fn main() {
//     let mut losses: Vec<f64> = vec![];

//     //#########################################################################
//     // Load data
//     //#########################################################################

//     // xor training data
//     let training_data = vec![
//         (vec![0.0, 0.0], vec![0.0]),
//         (vec![0.0, 1.0], vec![1.0]),
//         (vec![1.0, 0.0], vec![1.0]),
//         (vec![1.0, 1.0], vec![0.0]),
//     ];

//     //#########################################################################
//     // Create network
//     //#########################################################################

//     let mut network = Network::new(&[2, 2, 1]);
//     let learning_rate = 0.5;
//     let epochs = 10000;

//     //#########################################################################
//     // Training
//     //#########################################################################

//     for epoch in 0..epochs {
//         let mut loss_sum = 0.0;
//         for (ref input, ref target) in training_data.iter() {
//             let output = network.forward(input);
//             let loss = 0.5 * (output[0] - target[0]).powi(2); // mean squared error
//             loss_sum += loss;
//             // dl/dout (Gradient of loss for the network) = (output - target)
//             let dl_dout = vec![output[0] - target[0]];
//             network.backward(&dl_dout, learning_rate);
//         }
//         let loss_avg = loss_sum / training_data.len() as f64;
//         losses.push(loss_avg);
//         if epoch % 1000 == 0 {
//             println!("Epoch {}: avg loss = {}", epoch, loss_avg);
//         }
//     }

//     //#########################################################################
//     // Testing
//     //#########################################################################

//     for (input, target) in training_data.iter() {
//         let output = network.forward(input);
//         println!(
//             "Input: {:?}, target: {:?} output: {:?}",
//             input, target, output
//         );
//     }

//     //#########################################################################
//     // Visualization
//     //#########################################################################
//     let _ = draw_loss_over_time(losses);
// }
