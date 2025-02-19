use approx::{assert_abs_diff_eq, assert_relative_eq};
use puruspe::*;

const LAMBERT_W0_TABLE: [(f64, f64); 20] = [
    (1.00000000000000e-01, 9.12765271608623e-02),
    (2.00000000000000e-01, 1.68915973499110e-01),
    (5.00000000000000e-01, 3.51733711249196e-01),
    (1.00000000000000e+00, 5.67143290409784e-01),
    (1.50000000000000e+00, 7.25861357766226e-01),
    (2.00000000000000e+00, 8.52605502013725e-01),
    (2.50000000000000e+00, 9.58586356728703e-01),
    (3.00000000000000e+00, 1.04990889496404e+00),
    (4.00000000000000e+00, 1.20216787319704e+00),
    (5.00000000000000e+00, 1.32672466524220e+00),
    (1.00000000000000e+01, 1.74552800274070e+00),
    (2.00000000000000e+01, 2.20500327802406e+00),
    (5.00000000000000e+01, 2.86089017798221e+00),
    (2.50000000000000e-01, 2.03888354702240e-01),
    (7.50000000000000e-01, 4.69150210694988e-01),
    (1.00000000000000e-05, 9.99990000149997e-06),
    (1.00000000000000e-10, 9.99999999900000e-11),
    (1.00000000000000e+05, 9.28457142862211e+00),
    (1.00000000000000e+10, 2.00286854133050e+01),
    (1.00000000000000e+308, 7.02641362034107e+02),
];
const LAMBERT_WM1_TABLE: [(f64, f64); 13] = [
    (-1.62330466849397e-01, -2.87373297576420e+00),
    (-1.41318890794931e-02, -6.06123496778854e+00),
    (-1.78131724758206e-01, -2.72926392333150e+00),
    (-1.03843592031058e-01, -3.52465080303420e+00),
    (-3.64876111085733e-01, -1.13356447829203e+00),
    (-5.46821782149597e-02, -4.38423187070222e+00),
    (-3.63270038872334e-01, -1.16731536489767e+00),
    (-1.02171094066010e-02, -6.44736274096470e+00),
    (-2.37699099525656e-01, -2.24582072744266e+00),
    (-3.03767654679841e-01, -1.75258268173280e+00),
    (-1.00000000000000e-03, -9.11800647040274e+00),
    (-3.10000000000000e-05, -1.29420012897721e+01),
    (-1.00000000000000e-100, -2.35721158875685e+02),
];
const BRANCH_POINT: (f64, f64) = (-0.36787944117144232, -1.0);

#[test]
fn test_lambert_w0() {
    assert!(lambert_w0(-1.0).is_nan());
    assert!(sp_lambert_w0(-1.0).is_nan());
    assert_abs_diff_eq!(lambert_w0(BRANCH_POINT.0), BRANCH_POINT.1);
    assert_abs_diff_eq!(
        sp_lambert_w0(BRANCH_POINT.0),
        BRANCH_POINT.1,
        epsilon = 1e-7
    );
    for (x, y) in LAMBERT_W0_TABLE {
        assert_relative_eq!(lambert_w0(x), y, max_relative = 1e-14);
        assert_relative_eq!(sp_lambert_w0(x), y, max_relative = 1e-7);
    }
    assert_eq!(lambert_w0(f64::INFINITY), f64::INFINITY);
    assert_eq!(sp_lambert_w0(f64::INFINITY), f64::INFINITY);
    assert!(lambert_w0(f64::NAN).is_nan());
    assert!(sp_lambert_w0(f64::NAN).is_nan());
}

#[test]
fn test_lambert_wm1() {
    assert!(lambert_wm1(-1.0).is_nan());
    assert!(sp_lambert_wm1(-1.0).is_nan());
    assert_abs_diff_eq!(lambert_wm1(BRANCH_POINT.0), BRANCH_POINT.1);
    assert_abs_diff_eq!(
        sp_lambert_wm1(BRANCH_POINT.0),
        BRANCH_POINT.1,
        epsilon = 1e-7
    );
    for (x, y) in LAMBERT_WM1_TABLE {
        assert_relative_eq!(lambert_wm1(x), y, max_relative = 1e-14);
        assert_relative_eq!(sp_lambert_wm1(x), y, max_relative = 1e-7);
    }
    assert!(lambert_wm1(f64::NAN).is_nan());
    assert!(sp_lambert_wm1(f64::NAN).is_nan());
}
