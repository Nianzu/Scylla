use plotters::prelude::*;
use rand::Rng;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::time::SystemTime;
use std::vec;

struct Layer {
    weights: Vec<f32>,
    weight_width: usize,
    biases: Vec<f32>,

    last_input: Vec<f32>,
    last_output: Vec<f32>,
    dl_dinput: Vec<f32>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let weight_width = input_size;

        let weights = (0..(output_size * input_size)) // For each node in the current layer
            .map(|_| rng.random_range(-1.0..1.0))
            .collect::<Vec<f32>>();

        let biases = (0..output_size) // For each node
            .map(|_| rng.random_range(-1.0..1.0)) // Select a random number in our seed range
            .collect::<Vec<f32>>();

        let dl_dinput = vec![0.0; input_size];

        Self {
            weights,
            weight_width,
            biases,
            last_input: vec![],
            last_output: vec![],
            dl_dinput,
        }
    }

    // https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    fn sigmoid_derivative(sigmoid_x: f32) -> f32 {
        sigmoid_x * (1.0 - sigmoid_x)
    }

    fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        self.last_input = input.clone();
        let mut output = vec![];
        let num_neurons = self.biases.len();

        for i in 0..num_neurons {
            let start = i * self.weight_width;
            let end = start + self.weight_width;
            let neuron_weights = &self.weights[start..end];

            let z_i: f32 = neuron_weights
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
                + self.biases[i];

            let a_i = Self::sigmoid(z_i);
            output.push(a_i);
        }

        self.last_output = output.clone();
        output
    }

    fn backward(&mut self, dl_dout: &Vec<f32>, learning_rate: f32) -> &Vec<f32> {
        let num_neurons = self.biases.len();

        // dL/dz = dL/dout * sigmoid_derivative(z)
        let dl_dz: Vec<f32> = dl_dout
            .iter()
            .zip(self.last_output.iter())
            .map(|(&d, &a)| d * Self::sigmoid_derivative(a))
            .collect();

        // dL/db (Gradient for the bias) = dL/dz
        let dl_db = &dl_dz;

        // dL/dw (Gradient for the weight) = dL/dz * x for
        let input = &self.last_input;

        // Update weights and biases using gradient descent
        for i in 0..num_neurons {
            for j in 0..input.len() {
                unsafe {
                    self.weights[i * self.weight_width + j] -=
                        learning_rate * dl_dz.get_unchecked(i) * input.get_unchecked(j);
                }
            }
            unsafe {
                self.biases[i] -= learning_rate * dl_db.get_unchecked(i);
            }
        }

        // Compute and return dL/dx for previous layer
        // dL/dx = sum(dL/dz * w) for w in weights[i]
        for j in 0..input.len() {
            let mut sum = 0.0;
            for i in 0..num_neurons {
                unsafe {
                    sum += dl_dz.get_unchecked(i)
                        * self.weights.get_unchecked(i * self.weight_width + j);
                }
            }
            self.dl_dinput[j] = sum;
        }
        &self.dl_dinput
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

    fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output)
        }
        output
    }

    fn backward(&mut self, dl_dout: &Vec<f32>, learning_rate: f32) {
        let mut grad = dl_dout;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }
    }

    fn train(
        &mut self,
        dataset: &Vec<Vec<f32>>,
        labels: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> f32 {
        let mut loss_sum = 0.0;
        for i in 0..dataset.len() {
            let output = self.forward(&dataset[i]);
            let loss = rmse(&output, &labels[i]); // mean squared error
            loss_sum += loss;

            let dl_dout = network_gradient_loss(&output, &labels[i]);
            self.backward(&dl_dout, learning_rate);
        }
        loss_sum / dataset.len() as f32
    }

    fn validation_loss(&mut self, dataset: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
        let mut validation_loss_sum = 0.0;
        for i in 0..dataset.len() {
            let output = self.forward(&dataset[i]);
            let loss = rmse(&output, &labels[i]);
            validation_loss_sum += loss;
        }
        validation_loss_sum / dataset.len() as f32
    }

    fn accuracy(&mut self, dataset: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
        let mut accuracy = 0.0;
        for i in 0..dataset.len() {
            let output = self.forward(&dataset[i]);
            // https://stackoverflow.com/questions/53903318/what-is-the-idiomatic-way-to-get-the-index-of-a-maximum-or-minimum-floating-poin
            let predicted = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            let actual = labels[i]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            if predicted == actual {
                accuracy += 1.0;
            }
        }
        accuracy / dataset.len() as f32
    }
}

fn normalize_images(input: Vec<Vec<Vec<u8>>>) -> Vec<Vec<Vec<f32>>> {
    input
        .into_iter()
        .map(|vec1| {
            vec1.into_iter()
                .map(|vec2| vec2.into_iter().map(|item| item as f32 / 255.0).collect())
                .collect()
        })
        .collect()
}

fn flatten_images(input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<f32>> {
    input
        .into_iter()
        .map(|vec1| vec1.into_iter().flatten().collect())
        .collect()
}

fn flatten_labels(input: Vec<u8>) -> Vec<Vec<f32>> {
    let mut output = vec![];
    for item in input {
        let mut current_vec = vec![0.0; 10];
        current_vec[item as usize] = 1.0;
        output.push(current_vec);
    }
    output
}

// https://github.com/R34ll/mnist_rust
fn load_dataset_mnist_images(path: &str) -> Vec<Vec<Vec<f32>>> {
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

fn draw_mnist(image: Vec<Vec<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("plot.png", (840, 840)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let (width, height) = (image[0].len(), image.len());

    // Define the grid layout
    let cell_width: i32 = 840 / width as i32;
    let cell_height: i32 = 840 / height as i32;

    for (i, row) in image.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            let color = HSLColor(0.0, 0.0, value as f64);

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

fn draw_loss_over_time(
    losses: &Vec<f32>,
    validation_losses: &Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
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
    // .build_cartesian_2d(0.0..losses.len() as f32, (0.0..0.2).log_scale())?;

    // Draw the mesh for the chart
    chart.configure_mesh().draw()?; // Define the data points for the line

    // Plot the line using the data points
    chart
        .draw_series(LineSeries::new(
            losses
                .iter()
                .enumerate()
                .map(|(i, &loss)| (i as f64, loss as f64)),
            &RED,
        ))?
        .label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .draw_series(LineSeries::new(
            validation_losses
                .iter()
                .enumerate()
                .map(|(i, &loss)| (i as f64, loss as f64)),
            &GREEN,
        ))?
        .label("Validation Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

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

fn rmse(current: &Vec<f32>, target: &Vec<f32>) -> f32 {
    let n = current.len();
    let sum_squared_error: f32 = current
        .iter()
        .zip(target.iter())
        .map(|(c, t)| (c - t).powi(2))
        .sum();
    (sum_squared_error / n as f32).sqrt()
}

// dl/dout (Gradient of loss for the network) = (output - target)
fn network_gradient_loss(predictions: &Vec<f32>, target: &Vec<f32>) -> Vec<f32> {
    predictions
        .into_iter()
        .zip(target.clone().into_iter())
        .map(|(o, t)| (o - t))
        .collect()
}

fn main() {
    let mut losses: Vec<f32> = vec![];
    let mut validation_losses: Vec<f32> = vec![];
    let mut accuracies: Vec<f32> = vec![];

    //#########################################################################
    // Load data
    //#########################################################################

    // Training
    let dataset: Vec<Vec<Vec<f32>>> =
        load_dataset_mnist_images("datasets/mnist/train-images.idx3-ubyte");
    let labels: Vec<u8> = load_dataset_mnist_label("datasets/mnist/train-labels.idx1-ubyte");
    let flat_dataset: Vec<Vec<f32>> = flatten_images(dataset.clone());
    let flat_labels: Vec<Vec<f32>> = flatten_labels(labels.clone());

    // Validation
    let validation_dataset: Vec<Vec<Vec<f32>>> =
        load_dataset_mnist_images("datasets/mnist/t10k-images.idx3-ubyte");
    let validation_flat_dataset: Vec<Vec<f32>> = flatten_images(validation_dataset.clone());
    let validation_labels: Vec<u8> =
        load_dataset_mnist_label("datasets/mnist/t10k-labels.idx1-ubyte");
    let validation_flat_labels = flatten_labels(validation_labels.clone());

    //#########################################################################
    // Create network
    //#########################################################################

    let mut network = Network::new(&[784, 128, 10]);
    let learning_rate = 0.1;

    //#########################################################################
    // Training
    //#########################################################################
    let start = SystemTime::now();
    let mut dt = SystemTime::now().duration_since(start).expect("Error");
    let mut epoch = 0;
    let avg_epoch_time;
    let mut prev_validation_loss = 1.0;
    let mut validation_loss = 1.0;

    while dt.as_millis() < 10000_000 && validation_loss <= prev_validation_loss {
        let loss_avg = network.train(&flat_dataset, &flat_labels, learning_rate);
        losses.push(loss_avg);
        prev_validation_loss = validation_loss;
        validation_losses.push(validation_loss);

        validation_loss =
            network.validation_loss(&validation_flat_dataset, &validation_flat_labels);
        let accuracy = network.accuracy(&validation_flat_dataset, &validation_flat_labels);
        accuracies.push(accuracy);

        dt = SystemTime::now().duration_since(start).expect("Error");
        epoch += 1;
        println!(
            "Epoch: {}, Total time: {} | Training loss: {}, Validation loss: {}, Validation accuracy: {}",
            epoch,
            dt.as_millis(),
            loss_avg,
            validation_loss,
            accuracy,
        );
    }
    avg_epoch_time = dt.as_millis() as f32 / epoch as f32;

    //#########################################################################
    // Testing
    //#########################################################################
    let validation_loss =
        network.validation_loss(&validation_flat_dataset, &validation_flat_labels);
    println!("Validation loss: {}", validation_loss);

    //#########################################################################
    // Visualization
    //#########################################################################
    let _ = draw_loss_over_time(&losses, &validation_losses);
    let data = format!(
        "{:?},{},{}",
        validation_loss,
        avg_epoch_time,
        accuracies.last().unwrap()
    );
    fs::write("last_mnist_results", data).expect("Unable to write file");
}