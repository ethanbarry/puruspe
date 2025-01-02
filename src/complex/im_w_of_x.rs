//! An astounding number of coefficients!
//!
//!
//!

use super::{im_w_of_x_chebyshev_coeffs::*, INV_SQRT_PI};
use crate::utils::{frexp, sign};

/// Helper function for im_w_of_x(x).
fn cheb_interpolant(x: f64) -> f64 {
    const M: i32 = 6;
    let j0 = 0;
    let l0 = 0;
    let loff = (j0 + 1) * (1 << M) + l0;

    let (xm, je) = frexp(x);

    let ip = ((1 << (M + 1)) as f64 * xm) as i32;
    let lij = (je * (1 << M) + ip - loff) as i32;
    let t = ((1 << (M + 2)) as f64 * xm) - (1 + 2 * ip) as f64;

    let p_idx = lij * 8;
    let q_idx = lij * 2;

    assert!(
        p_idx >= 0 && q_idx >= 0,
        "Negative indices not permitted in Rust."
    );

    ((((((((CHEBYSHEV_COEFFS_1[(p_idx + 7) as usize] * t
        + CHEBYSHEV_COEFFS_1[(p_idx + 6) as usize])
        * t
        + CHEBYSHEV_COEFFS_1[(p_idx + 5) as usize])
        * t
        + CHEBYSHEV_COEFFS_1[(p_idx + 4) as usize])
        * t
        + CHEBYSHEV_COEFFS_1[(p_idx + 3) as usize])
        * t
        + CHEBYSHEV_COEFFS_1[(p_idx + 2) as usize])
        * t
        + CHEBYSHEV_COEFFS_1[(p_idx + 1) as usize])
        * t
        + CHEBYSHEV_COEFFS_1[p_idx as usize])
        * t
        + CHEBYSHEV_COEFFS_0[(q_idx + 1) as usize])
        * t
        + CHEBYSHEV_COEFFS_0[q_idx as usize]
}

/// The imaginary part of the Faddeeva function w(x) on a real input x,
/// which is equal to the scaled Dawson integral.
///
/// ### Definition:
/// 2 * Dawson(x) / sqrt(pi) &nbsp; &nbsp; &nbsp; *(result is imaginary)*
///
/// ### Input:
/// - x: A real number
///
/// ### Output:
/// The imaginary component of the Faddeeva function evaluated at the
/// complex number x + 0𝑖, or ℑ(w(x + 0𝑖)).
///
/// ### Notes on Implementation:
/// This is a Rust port of the `im_w_of_x` routine found in [`libcerf`](https://jugit.fz-juelich.de/mlz/libcerf),
/// which is under the MIT License.
///
/// It uses the following methods:
/// - Asymptotic expansion for large |x|
/// - Chebyshev polynomial expansion for medium |x|
/// - Maclaurin series for small |x|
pub fn im_w_of_x(x: f64) -> f64 {
    let xabs = x.abs();

    if xabs < 0.51 {
        // The Maclaurin series (2/sqrt(pi)) * (x - 2/3 x^3  + 4/15 x^5  - 8/105 x^7 ...)

        let x_squared = x * x;

        if xabs < 0.083 {
            if xabs < 0.003 {
                return ((((-0.085971746064420005629) * x_squared // x^7
                          + 0.30090111122547001970) * x_squared // x^5
                         - 0.75225277806367504925) * x_squared // x^3
                        + 1.1283791670955125739)
                    * x;
            }

            return (((((((0.00053440090793734269229) * x_squared // x^13
                         - 0.0034736059015927275001) * x_squared // x^11
                        + 0.019104832458760001251) * x_squared // x^9
                       - 0.085971746064420005629) * x_squared // x^7
                      + 0.30090111122547001970) * x_squared // x^5
                     - 0.75225277806367504925) * x_squared // x^3
                    + 1.1283791670955125739)
                * x;
        }

        if xabs < 0.272 {
            return ((((((((((-8.82395720020380130481012927e-7) * x_squared // x^19
                            + 8.38275934019361123956e-6) * x_squared // x^17
                           - 7.1253454391645686483238e-5) * x_squared // x^15
                          + 0.00053440090793734269229) * x_squared // x^13
                         - 0.0034736059015927275001) * x_squared // x^11
                        + 0.019104832458760001251) * x_squared // x^9
                       - 0.085971746064420005629) * x_squared // x^7
                      + 0.30090111122547001970) * x_squared // x^5
                     - 0.75225277806367504925) * x_squared // x^3
                    + 1.1283791670955125739)
                * x;
        }

        return (((((((((((((5.8461000084165966602290712e-10) * x_squared // x^25
                           - 7.30762501052074563638866034e-9) * x_squared // x^23
                          + 8.40376876209885782941868884e-8) * x_squared // x^21
                         - 8.82395720020380130481012927e-7) * x_squared // x^19
                        + 8.38275934019361123956e-6) * x_squared // x^17
                       - 7.1253454391645686483238e-5) * x_squared // x^15
                      + 0.00053440090793734269229) * x_squared // x^13
                     - 0.0034736059015927275001) * x_squared // x^11
                    + 0.019104832458760001251) * x_squared // x^9
                   - 0.085971746064420005629) * x_squared // x^7
                  + 0.30090111122547001970) * x_squared // x^5
                 - 0.75225277806367504925) * x_squared // x^3
                + 1.1283791670955125739)
            * x;
    }

    if xabs < 12. {
        return sign(cheb_interpolant(xabs), x); // Medium |x|: Use Chebyshev interpolants.
    }

    // Use asymptotic expansion up to N = 0, 3, 6, or 10
    //
    // With N=15 or 20 we could extend the range down to 7.73 or 6.72,
    // but we expect Chebyshev to be faster.
    //
    // Coefficients are a_0 = 1/sqrt(pi), a_N = (2N-1)!!/2^N/sqrt(pi).

    let r = 1. / x;
    let r_squared = 1. / (x * x);

    if xabs < 150. {
        if xabs < 23.2 {
            return (((((((((((3.6073371500083758e+05) * r_squared
                + 3.7971970000088164e+04)
                * r_squared
                + 4.4672905882456671e+03)
                * r_squared
                + 5.9563874509942218e+02)
                * r_squared
                + 9.1636730015295726e+01)
                * r_squared
                + 1.6661223639144676e+01)
                * r_squared
                + 3.7024941420321507e+00)
                * r_squared
                + 1.0578554691520430e+00)
                * r_squared
                + 4.2314218766081724e-01)
                * r_squared
                + 2.8209479177387814e-01)
                * r_squared
                + 5.6418958354775628e-01)
                * r;
        }

        return (((((((9.1636730015295726e+01) * r_squared + 1.6661223639144676e+01)
            * r_squared
            + 3.7024941420321507e+00)
            * r_squared
            + 1.0578554691520430e+00)
            * r_squared
            + 4.2314218766081724e-01)
            * r_squared
            + 2.8209479177387814e-01)
            * r_squared
            + 5.6418958354775628e-01)
            * r;
    }

    if xabs < 6.9e7 {
        return ((((1.0578554691520430e+00) * r_squared + 4.2314218766081724e-01) * r_squared
            + 2.8209479177387814e-01)
            * r_squared
            + 5.6418958354775628e-01)
            * r;
    }

    // 1-term expansion, important to avoid overflow
    return INV_SQRT_PI / x;
}
