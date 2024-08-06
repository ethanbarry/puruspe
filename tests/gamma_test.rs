use approx::assert_abs_diff_eq;
use puruspe::{
    ln_gamma,
};

const LN_GAMMA_TABLE: [(f64, f64); 19] = [
    (1.0000000000000001e-01, 2.2527126517342060e+00),
    (2.0000000000000001e-01, 1.5240638224307845e+00),
    (5.0000000000000000e-01, 5.7236494292469997e-01),
    (1.0000000000000000e+00, 0.0000000000000000e+00),
    (1.5000000000000000e+00, -1.2078223763524526e-01),
    (2.0000000000000000e+00, 0.0000000000000000e+00),
    (2.5000000000000000e+00, 2.8468287047291918e-01),
    (3.0000000000000000e+00, 6.9314718055994529e-01),
    (4.0000000000000000e+00, 1.7917594692280550e+00),
    (5.0000000000000000e+00, 3.1780538303479458e+00),
    (1.0000000000000000e+01, 1.2801827480081469e+01),
    (2.0000000000000000e+01, 3.9339884187199495e+01),
    (5.0000000000000000e+01, 1.4456574394634487e+02),
    (2.5000000000000000e-01, 1.2880225246980774e+00),
    (7.5000000000000000e-01, 2.0328095143129538e-01),
    (1.0000000000000001e-05, 1.1512919692895826e+01),
    (1.0000000000000000e-10, 2.3025850929882733e+01),
    (1.0000000000000000e+05, 1.0512877089736569e+06),
    (1.0000000000000000e+10, 2.2025850928881058e+11),
];

#[test]
fn test_ln_gamma() {
    for (x, y) in LN_GAMMA_TABLE {
        let eps = if x <= 1e-6 {
            1e-6
        } else if x >= 1e+3 {
            x * 1e-13
        } else {
            1e-10
        };
        assert_abs_diff_eq!(ln_gamma(x), y, epsilon = eps);
    }
}
