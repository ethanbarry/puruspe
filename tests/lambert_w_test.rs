use puruspe::*;
use approx::assert_abs_diff_eq;

const LAMBERT_W0_TABLE: [(f64, f64); 21] = [
    (-3.6787944117144233e-01, -1.0000000000000000e+00),
    (1.0000000000000001e-01, 9.1276527160862264e-02),
    (2.0000000000000001e-01, 1.6891597349910958e-01),
    (5.0000000000000000e-01, 3.5173371124919584e-01),
    (1.0000000000000000e+00, 5.6714329040978384e-01),
    (1.5000000000000000e+00, 7.2586135776622629e-01),
    (2.0000000000000000e+00, 8.5260550201372542e-01),
    (2.5000000000000000e+00, 9.5858635672870285e-01),
    (3.0000000000000000e+00, 1.0499088949640398e+00),
    (4.0000000000000000e+00, 1.2021678731970429e+00),
    (5.0000000000000000e+00, 1.3267246652422002e+00),
    (1.0000000000000000e+01, 1.7455280027406994e+00),
    (2.0000000000000000e+01, 2.2050032780240598e+00),
    (5.0000000000000000e+01, 2.8608901779822107e+00),
    (2.5000000000000000e-01, 2.0388835470224018e-01),
    (7.5000000000000000e-01, 4.6915021069498819e-01),
    (1.0000000000000001e-05, 9.9999000014999740e-06),
    (1.0000000000000000e-10, 9.9999999989999997e-11),
    (1.0000000000000000e+05, 9.2845714286221082e+00),
    (1.0000000000000000e+10, 2.0028685413304952e+01),
    (1.0000000000000000e+308, 7.0264136203410681e+02),
];

const LAMBERT_WM1_TABLE: [(f64, f64); 13] = [
    (-1.6233046684939709e-01, -2.8737329757642014e+00),
    (-1.4131889079493100e-02, -6.0612349677885398e+00),
    (-1.7813172475820591e-01, -2.7292639233315006e+00),
    (-1.0384359203105779e-01, -3.5246508030341959e+00),
    (-3.6487611108573298e-01, -1.1335644782920300e+00),
    (-5.4682178214959701e-02, -4.3842318707022221e+00),
    (-3.6327003887233378e-01, -1.1673153648976677e+00),
    (-1.0217109406601000e-02, -6.4473627409647047e+00),
    (-2.3769909952565629e-01, -2.2458207274426640e+00),
    (-3.0376765467984079e-01, -1.7525826817328027e+00),
    (-1.0000000000000000e-03, -9.1180064704027401e+00),
    (-3.1000000000000001e-05, -1.2942001289772060e+01),
    (-1.0000000000000000e-100, -2.3572115887568532e+02),
];

#[test]
fn test_lambert_w0() {
    assert!(lambert_w0(-1.0).is_nan());
    assert!(sp_lambert_w0(-1.0).is_nan());
    for (x, y) in LAMBERT_W0_TABLE {
        // Less accurate in absolute difference for larger values
        let small_epsilon = if x <= 1e8 {
            1e-14
        } else if x <= 1e20 {
            1e-13
        } else {
            1e-12
        };
        let large_epsilon = small_epsilon * 1e8;
        assert_abs_diff_eq!(lambert_w0(x), y, epsilon = small_epsilon);
        assert_abs_diff_eq!(sp_lambert_w0(x), y, epsilon = large_epsilon);
    }
}

#[test]
fn test_lambert_wm1() {
    assert!(lambert_wm1(-1.0).is_nan());
    assert!(sp_lambert_wm1(-1.0).is_nan());
    for (x, y) in LAMBERT_WM1_TABLE {
        let small_epsilon = 1e-14;
        // 24 bit version is slightly less accurate for very very small values
        let big_epsilon = small_epsilon * if x < -1e-10 { 1e8 } else { 1e9 };
        println!("{big_epsilon}");
        assert_abs_diff_eq!(lambert_wm1(x), y, epsilon = small_epsilon);
        assert_abs_diff_eq!(sp_lambert_wm1(x), y, epsilon = big_epsilon);
    }
}
