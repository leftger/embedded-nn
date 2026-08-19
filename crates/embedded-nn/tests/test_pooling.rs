use embedded_nn::{
    Activation, Dims, PoolParams, Tile,
    pooling::{avg_pool_s8, avg_pool_s16, max_pool_s8, max_pool_s16},
};

#[test]
fn test_max_pool_s8_variations() {
    let pool_params = PoolParams {
        stride: Tile::new(2, 2),
        padding: Tile::new(0, 0),
        activation: Activation::new(-10, 10),
    };
    let filter_dims = Tile::new(2, 2);

    let input_dims = Dims::new(1, 4, 4, 1);
    let input = [
        -20i8, -15i8, 5i8, 6i8, -10i8, -5i8, 7i8, 8i8, 9i8, 10i8, 13i8, 14i8, 11i8, 12i8, 15i8,
        16i8,
    ];
    let output_dims = Dims::new(1, 2, 2, 1);
    let mut output = [0i8; 4];

    max_pool_s8(
        &pool_params,
        &filter_dims,
        &input_dims,
        &input,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Top-left block max = -5
    // Top-right block max = 8
    // Bottom-left block max = 12 -> clamped to 10
    // Bottom-right block max = 16 -> clamped to 10
    assert_eq!(output, [-5, 8, 10, 10]);
}

#[test]
fn test_avg_pool_s8_negative_values_and_padding() {
    let pool_params = PoolParams {
        stride: Tile::new(1, 1),
        padding: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let filter_dims = Tile::new(3, 3);

    let input_dims = Dims::new(1, 2, 2, 1);
    let input = [-10i8, -20i8, -30i8, -40i8];
    let output_dims = Dims::new(1, 2, 2, 1);
    let mut output = [0i8; 4];

    avg_pool_s8(
        &pool_params,
        &filter_dims,
        &input_dims,
        &input,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Top-left (0,0): covers 2x2 elements (-10, -20, -30, -40) -> sum = -100, count = 4
    // avg = -100 / 4 = -25
    assert_eq!(output[0], -25);
}

#[test]
fn test_max_pool_s16_multi_channel() {
    let pool_params = PoolParams {
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        activation: Activation::int16_unconstrained(),
    };
    let filter_dims = Tile::new(2, 2);

    let input_dims = Dims::new(1, 2, 2, 2); // N=1, H=2, W=2, C=2
    let input = [
        100i16, -200i16, // (0,0)
        300i16, -400i16, // (0,1)
        500i16, -600i16, // (1,0)
        700i16, -100i16, // (1,1)
    ];
    let output_dims = Dims::new(1, 1, 1, 2);
    let mut output = [0i16; 2];

    max_pool_s16(
        &pool_params,
        &filter_dims,
        &input_dims,
        &input,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Ch 0 max: max(100, 300, 500, 700) = 700
    // Ch 1 max: max(-200, -400, -600, -100) = -100
    assert_eq!(output, [700, -100]);
}

#[test]
fn test_avg_pool_s16_large_sums() {
    let pool_params = PoolParams {
        stride: Tile::new(2, 2),
        padding: Tile::new(0, 0),
        activation: Activation::int16_unconstrained(),
    };
    let filter_dims = Tile::new(2, 2);

    let input_dims = Dims::new(1, 2, 2, 1);
    let input = [30000i16, 30000i16, 30000i16, 30000i16];
    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0i16; 1];

    avg_pool_s16(
        &pool_params,
        &filter_dims,
        &input_dims,
        &input,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // sum = 120000 -> count = 4 -> avg = 30000
    assert_eq!(output[0], 30000);
}
