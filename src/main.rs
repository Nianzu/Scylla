use plotters::prelude::*;
use plotters::style::Color;
use rand::Rng;
use rschess::Move;
use rschess::{Board, Color as chessColor, PieceType};
use std::fs;
use std::fs::File;
use std::io;
use std::io::{prelude::*, BufReader};
use std::time::SystemTime;
use std::vec;
mod neural_network;

struct Scylla {
    pawns: Vec<i8>,
    bishops: Vec<i8>,
    knights: Vec<i8>,
    rooks: Vec<i8>,
    queens: Vec<i8>,
    kings: Vec<i8>,
    network_piece_selector: neural_network::Network,
    network_pawns: neural_network::Network,
    network_bishops: neural_network::Network,
    network_knights: neural_network::Network,
    network_rooks: neural_network::Network,
    network_queens: neural_network::Network,
    network_kings: neural_network::Network,
}

impl Scylla {
    fn new(
        network_piece_selector: neural_network::Network,
        network_pawns: neural_network::Network,
        network_bishops: neural_network::Network,
        network_knights: neural_network::Network,
        network_rooks: neural_network::Network,
        network_queens: neural_network::Network,
        network_kings: neural_network::Network,
    ) -> Scylla {
        Scylla {
            pawns: vec![0; 64],
            bishops: vec![0; 64],
            knights: vec![0; 64],
            rooks: vec![0; 64],
            queens: vec![0; 64],
            kings: vec![0; 64],
            network_piece_selector,
            network_pawns,
            network_bishops,
            network_knights,
            network_rooks,
            network_queens,
            network_kings,
        }
    }

    fn best_move(mut self, game_state: &Board) -> String {
        let mut index = 0;
        for j in ('1'..='8').rev() {
            for i in 'a'..='h' {
                if !game_state.occupant_of_square(i, j).unwrap().is_none() {
                    let piece_value = if game_state
                        .occupant_of_square(i, j)
                        .unwrap()
                        .unwrap()
                        .color()
                        == chessColor::White
                    {
                        1
                    } else {
                        -1
                    };
                    let piece_type = game_state
                        .occupant_of_square(i, j)
                        .unwrap()
                        .unwrap()
                        .piece_type();
                    match piece_type {
                        PieceType::P => self.pawns[index] = piece_value,
                        PieceType::B => self.bishops[index] = piece_value,
                        PieceType::N => self.knights[index] = piece_value,
                        PieceType::R => self.rooks[index] = piece_value,
                        PieceType::Q => self.queens[index] = piece_value,
                        PieceType::K => self.kings[index] = piece_value,
                    }
                }
                index += 1;
            }
        }

        let vecs = vec![
            &self.pawns,
            &self.bishops,
            &self.knights,
            &self.rooks,
            &self.queens,
            &self.kings,
        ];

        let result = combine_vecs(vecs);
        let pred_piece_sel = self.network_piece_selector.forward(&result);
        let pred_pawns = self.network_pawns.forward(&result);
        let pred_bishops = self.network_bishops.forward(&result);
        let pred_knights = self.network_knights.forward(&result);
        let pred_rooks = self.network_rooks.forward(&result);
        let pred_queens = self.network_queens.forward(&result);
        let pred_kings = self.network_kings.forward(&result);
        let mut best_move_score = 0.0;
        let mut best_move_uci = "".to_string();
        for rank in ('1'..='8').rev() {
            for file in 'a'..='h' {
                let file_int = file as usize - 97;
                let rank_int = rank as usize - 49;
                let (dst_score, dst_uci) = get_best_move_and_score(
                    rank,
                    file,
                    &game_state,
                    &pred_pawns,
                    &pred_bishops,
                    &pred_knights,
                    &pred_rooks,
                    &pred_queens,
                    &pred_kings,
                );
                let move_score = pred_piece_sel[file_int + ((7 - rank_int) * 8)] * dst_score;
                if move_score > best_move_score {
                    best_move_score = move_score;
                    best_move_uci =
                        "".to_string() + &file.to_string() + &rank.to_string() + &dst_uci;
                }
            }
        }
        best_move_uci
    }
}

fn load_scylla_csv(path: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut train_data = Vec::new();
    let mut train_labels = Vec::new();
    let mut validation_data = Vec::new();
    let mut validation_labels = Vec::new();

    let file = File::open(path).expect("File find no");
    let reader = BufReader::new(file);
    let mut rng = rand::rng();

    let mut is_data = true;
    let mut is_training: bool = true;
    for line in reader.lines() {
        let processed_line = line.expect("REASON");
        let mut processed_line_2 = processed_line.split(",").collect::<Vec<_>>();
        processed_line_2.retain(|a| !a.is_empty());

        let line_vec: Vec<f32> = processed_line_2
            .into_iter()
            .map(|a| {
                a.trim()
                    .parse::<f32>()
                    .expect("Error parsing value as float")
            })
            .collect::<Vec<_>>();
        if is_data {
            is_training = rng.random_range(1..=10) != 1;
            match is_training {
                true => {
                    train_data.push(line_vec);
                }
                false => {
                    validation_data.push(line_vec);
                }
            }
        } else {
            match is_training {
                true => {
                    train_labels.push(line_vec);
                }
                false => {
                    validation_labels.push(line_vec);
                }
            }
        }
        is_data = !is_data;
    }
    println!("Loaded data file \"{}\"", path);
    println!(
        "(Train, Validation) dataset size: ({}, {})",
        train_data.len(),
        validation_data.len()
    );

    (train_data, train_labels, validation_data, validation_labels)
}

fn draw_loss_over_time(
    name: &str,
    losses: &Vec<f32>,
    validation_losses: &Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a drawing area for the plot
    let file_path = name.to_owned() + ".png";
    let root = BitMapBackend::new(&file_path, (640, 480)).into_drawing_area();
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

// ! AI Generated
fn combine_vecs(vecs: Vec<&Vec<i8>>) -> Vec<f32> {
    let mut combined: Vec<f32> = Vec::new();

    for vec in vecs {
        combined.extend(vec.into_iter().map(|x| *x as f32));
    }

    combined
}

fn get_best_move_and_score(
    rank: char,
    file: char,
    board: &Board,
    pred_pawns: &Vec<f32>,
    pred_bishops: &Vec<f32>,
    pred_knights: &Vec<f32>,
    pred_rooks: &Vec<f32>,
    pred_queens: &Vec<f32>,
    pred_kings: &Vec<f32>,
) -> (f32, String) {
    if board.occupant_of_square(file, rank).unwrap().is_none() {
        return (0.0, "".to_owned());
    }
    if board
        .occupant_of_square(file, rank)
        .unwrap()
        .unwrap()
        .color()
        == chessColor::Black
    {
        return (0.0, "".to_owned());
    }
    let pos_uci_start = format!("{}{}", file, rank);
    let mut moves_from_pos: Vec<String> = vec![];
    for mv in board.gen_legal_moves() {
        let move_str = mv.to_uci();

        if move_str.starts_with(&pos_uci_start) {
            let split_pos = move_str.char_indices().nth_back(1).unwrap().0;
            moves_from_pos.push(move_str[split_pos..].to_string());
        }
    }

    let mut best_move_score = 0.0;
    let mut best_move_uci = "".to_owned();
    for m in moves_from_pos {
        let file_int = m.chars().nth(0).unwrap() as usize - 97;
        let rank_int = m.chars().nth(1).unwrap() as usize - 49;
        let index = file_int + ((7 - rank_int) * 8);
        let score = match board
            .occupant_of_square(file, rank)
            .unwrap()
            .unwrap()
            .piece_type()
        {
            PieceType::P => pred_pawns[index],
            PieceType::B => pred_bishops[index],
            PieceType::N => pred_knights[index],
            PieceType::R => pred_rooks[index],
            PieceType::Q => pred_queens[index],
            PieceType::K => pred_kings[index],
        };
        if score > best_move_score {
            best_move_score = score;
            best_move_uci = m;
        }
    }

    return (best_move_score, best_move_uci);
}

fn main() {
    let banner = "                                                     _:_    \n".to_owned()
        + "                                                    '-.-'   \n"
        + "                                           ()      __.'.__  \n"
        + "                                        .-:--:-.  |_______| \n"
        + "                                 ()      \\____/    \\=====/      ()          \n"
        + "                                 /\\      {====}     )___(       /\\               \n"
        + "                      (\\=,      //\\\\      )__(     /_____\\     //\\\\     (\\=,        \n"
        + "      __    |'-'-'|  //  .\\    (    )    /____\\     |   |     (    )   //  .\\   |'-'-'|    __  \n"
        + "     /  \\   |_____| (( \\_  \\    )__(      |  |      |   |      )__(   (( \\_  \\  |_____|   /  \\  \n"
        + "     \\__/    |===|   ))  `\\_)  /____\\     |  |      |   |     /____\\   ))  `\\_)  |===|    \\__/  \n"
        + "    /____\\   |   |  (/     \\    |  |      |  |      |   |      |  |   (/     \\   |   |   /____\\   \n"
        + "     |  |    |   |   | _.-'|    |  |      |  |      |   |      |  |    | _.-'|   |   |    |  |  \n"
        + "     |__|    )___(    )___(    /____\\    /____\\    /_____\\    /____\\    )___(    )___(    |__| \n"
        + "    (====)  (=====)  (=====)  (======)  (======)  (=======)  (======)  (=====)  (=====)  (====)\n"
        + "    }===={  }====={  }====={  }======{  }======{  }======={  }======{  }====={  }====={  }===={\n"
        + "   (______)(_______)(_______)(________)(________)(_________)(________)(_______)(_______)(______)\n";
    let text = "                           .d8888b.                    888 888           \n"
        .to_owned()
        + "                           d88P  Y88b                   888 888          \n"
        + "                           Y88b.                        888 888          \n"
        + "                            \"Y888b.    .d8888b 888  888 888 888  8888b.  \n"
        + "                               \"Y88b. d88P\"    888  888 888 888     \"88b \n"
        + "                                 \"888 888      888  888 888 888 .d888888 \n"
        + "                           Y88b  d88P Y88b.    Y88b 888 888 888 888  888 \n"
        + "                            \"Y8888P\"   \"Y8888P  \"Y88888 888 888 \"Y888888 \n"
        + "                                                    888                  \n"
        + "                                               Y8b d88P                  \n"
        + "                                                \"Y88P\"                   \n\n"
        + "                                  Ascii Art credit: Joan G. Stark\n"
        + "                                       Software by: Nico Zucca\n\n\n\n";
    let network_names = vec![
        "piece_selector",
        "pawn",
        "bishop",
        "knight",
        "rook",
        "queen",
        "king",
    ];
    let trained_networks: Vec<String> = fs::read_dir("trained_network")
        .expect("Unable to read trained_networks")
        .map(|a| a.unwrap().path().to_str().unwrap().to_owned())
        .collect();
    let mut networks: Vec<neural_network::Network> = vec![];

    for network_name in network_names {
        let mut losses: Vec<f32> = vec![];
        let mut validation_losses: Vec<f32> = vec![];
        let mut accuracies: Vec<f32> = vec![];
        //#########################################################################
        // Load data
        //#########################################################################

        let network_path = "trained_network/".to_owned() + network_name + ".bin";
        if trained_networks.contains(&network_path) {
            networks.push(neural_network::Network::load(&network_path));
            println!("Loaded Network: {}", network_name);
        } else {
            let (flat_dataset, flat_labels, validation_flat_dataset, validation_flat_labels) =
                load_scylla_csv(&("datasets/chess_2000/".to_owned() + network_name + ".csv"));

            println!("Training Network: {}", network_name);
            //#########################################################################
            // Create network
            //#########################################################################

            let mut network = neural_network::Network::new(network_name, &[384, 128, 64]);
            let learning_rate = 0.01;

            //#########################################################################
            // Training
            //#########################################################################
            let start = SystemTime::now();
            let mut dt = SystemTime::now().duration_since(start).expect("Error");
            let mut epoch = 0;
            let mut prev_validation_loss = 1.0;
            let mut validation_loss = 1.0;

            while dt.as_secs() < 100_000 && validation_loss <= prev_validation_loss && epoch < 200 {
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
                print!(
                    "\rEpoch: {:3}, Total time: {:5}s | Training loss: {:8}, Validation loss: {:8}, Validation accuracy: {:8}",
                    epoch,
                    dt.as_secs(),
                    loss_avg,
                    validation_loss,
                    accuracy,
                );
                std::io::stdout().flush().unwrap();
                network.save();
            }
            println!("");
            networks.push(network);

            //#########################################################################
            // Visualization
            //#########################################################################
            let _ = draw_loss_over_time(network_name, &losses, &validation_losses);

            //#########################################################################
            // Testing
            //#########################################################################
            let network_length = networks.len();
            let validation_loss = networks[network_length - 1]
                .accuracy(&validation_flat_dataset, &validation_flat_labels);
            println!("Accuracy: {}", validation_loss);
        }
    }
    println!("{}", banner);
    println!("{}", text);
    let mut game_state = Board::default();
    let mut game_over = false;
    while !game_over {
        //#########################################################################
        // Scylla
        //#########################################################################
        let scylla = Scylla::new(
            networks[0].clone(),
            networks[1].clone(),
            networks[2].clone(),
            networks[3].clone(),
            networks[4].clone(),
            networks[5].clone(),
            networks[6].clone(),
        );

        let best_move_uci = scylla.best_move(&game_state);

        println!("Scylla plays {}", best_move_uci);
        game_state
            .make_move(Move::from_uci(&best_move_uci).unwrap())
            .unwrap();

        if game_state.is_checkmate() {
            println!("Scylla wins by checkmate!");
            game_over = true;
        }

        //#########################################################################
        // Human
        //#########################################################################
        println!("{}", game_state);
        let mut human_made_legal_move = false;
        while !human_made_legal_move {
            print!("Your move: ");
            std::io::stdout().flush().unwrap();
            let mut move_string = String::new();
            io::stdin()
                .read_line(&mut move_string)
                .expect("Error reading input");
            match game_state.make_move_uci(move_string.trim()) {
                Ok(()) => {
                    human_made_legal_move = true;
                }
                Err(_) => {
                    println!("BAD MOVE")
                }
            }
        }

        if game_state.is_checkmate() {
            println!("You win by checkmate!");
            game_over = true;
        }
    }
}
