use approx::assert_abs_diff_eq;
use puruspe::{
    ln_gamma,
    gamma,
    gammp,
    gammq,
    invgammp
};

const LN_GAMMA_TABLE: [(f64, f64); 19] = [
    (1.00000000000000e-01, 2.25271265173421e+00),
    (2.00000000000000e-01, 1.52406382243078e+00),
    (5.00000000000000e-01, 5.72364942924700e-01),
    (1.00000000000000e+00, 0.00000000000000e+00),
    (1.50000000000000e+00, -1.20782237635245e-01),
    (2.00000000000000e+00, 0.00000000000000e+00),
    (2.50000000000000e+00, 2.84682870472919e-01),
    (3.00000000000000e+00, 6.93147180559945e-01),
    (4.00000000000000e+00, 1.79175946922805e+00),
    (5.00000000000000e+00, 3.17805383034795e+00),
    (1.00000000000000e+01, 1.28018274800815e+01),
    (2.00000000000000e+01, 3.93398841871995e+01),
    (5.00000000000000e+01, 1.44565743946345e+02),
    (2.50000000000000e-01, 1.28802252469808e+00),
    (7.50000000000000e-01, 2.03280951431295e-01),
    (1.00000000000000e-05, 1.15129196928958e+01),
    (1.00000000000000e-10, 2.30258509298827e+01),
    (1.00000000000000e+05, 1.05128770897366e+06),
    (1.00000000000000e+10, 2.20258509288811e+11),
];

#[test]
fn test_ln_gamma() {
    for (x, y) in LN_GAMMA_TABLE {
        let eps = 1e-10 + 1e-8 * ln_gamma(x).max(y).abs();
        if (ln_gamma(x) - y).abs() > eps {
            dbg!(x, ln_gamma(x), y);
        }
        assert_abs_diff_eq!(ln_gamma(x), y, epsilon = eps);
    }
}

const GAMMA_TABLE: [(f64, f64); 21] = [
    (-1.50000000000000e+00, 2.36327180120735e+00),
    (-5.00000000000000e-01, -3.54490770181103e+00),
    (1.00000000000000e-01, 9.51350769866873e+00),
    (2.00000000000000e-01, 4.59084371199880e+00),
    (5.00000000000000e-01, 1.77245385090552e+00),
    (1.00000000000000e+00, 1.00000000000000e+00),
    (1.50000000000000e+00, 8.86226925452758e-01),
    (2.00000000000000e+00, 1.00000000000000e+00),
    (2.50000000000000e+00, 1.32934038817914e+00),
    (3.00000000000000e+00, 2.00000000000000e+00),
    (4.00000000000000e+00, 6.00000000000000e+00),
    (5.00000000000000e+00, 2.40000000000000e+01),
    (1.00000000000000e+01, 3.62880000000000e+05),
    (2.00000000000000e+01, 1.21645100408832e+17),
    (5.00000000000000e+01, 6.08281864034268e+62),
    (2.50000000000000e-01, 3.62560990822191e+00),
    (7.50000000000000e-01, 1.22541670246518e+00),
    (1.00000000000000e-05, 9.99994227942255e+04),
    (1.00000000000000e-10, 9.99999999942278e+09),
    (1.00000000000000e+02, 9.33262154439442e+155),
    (1.70000000000000e+02, 4.26906800900471e+304),
];

#[test]
fn test_gamma() {
    for (x, y) in GAMMA_TABLE {
        let eps = f64::EPSILON + 1e-10 * gamma(x).max(y).abs();
        if (gamma(x) - y).abs() > eps {
            dbg!(x, gamma(x), y);
        }
        assert_abs_diff_eq!(gamma(x), y, epsilon = eps);
    }
}

const GAMMP_TABLE: [(f64, f64, f64); 42] = [
    (1.00000000000000e-01, 1.00000000000000e-01, 8.27551759585850e-01),
    (1.00000000000000e-01, 5.00000000000000e-01, 9.41402445890133e-01),
    (1.00000000000000e-01, 1.00000000000000e+00, 9.75872656273672e-01),
    (1.00000000000000e-01, 2.00000000000000e+00, 9.94326176020188e-01),
    (1.00000000000000e-01, 5.00000000000000e+00, 9.99856061034153e-01),
    (1.00000000000000e-01, 1.00000000000000e+01, 9.99999445201428e-01),
    (1.00000000000000e-01, 2.00000000000000e+01, 9.99999999985986e-01),
    (5.00000000000000e-01, 1.00000000000000e-01, 3.45279153981423e-01),
    (5.00000000000000e-01, 5.00000000000000e-01, 6.82689492137086e-01),
    (5.00000000000000e-01, 1.00000000000000e+00, 8.42700792949715e-01),
    (5.00000000000000e-01, 2.00000000000000e+00, 9.54499736103641e-01),
    (5.00000000000000e-01, 5.00000000000000e+00, 9.98434597741997e-01),
    (5.00000000000000e-01, 1.00000000000000e+01, 9.99992255783569e-01),
    (5.00000000000000e-01, 2.00000000000000e+01, 9.99999999746037e-01),
    (1.00000000000000e+00, 1.00000000000000e-01, 9.51625819640404e-02),
    (1.00000000000000e+00, 5.00000000000000e-01, 3.93469340287367e-01),
    (1.00000000000000e+00, 1.00000000000000e+00, 6.32120558828558e-01),
    (1.00000000000000e+00, 2.00000000000000e+00, 8.64664716763387e-01),
    (1.00000000000000e+00, 5.00000000000000e+00, 9.93262053000915e-01),
    (1.00000000000000e+00, 1.00000000000000e+01, 9.99954600070238e-01),
    (1.00000000000000e+00, 2.00000000000000e+01, 9.99999997938846e-01),
    (2.00000000000000e+00, 1.00000000000000e-01, 4.67884016044447e-03),
    (2.00000000000000e+00, 5.00000000000000e-01, 9.02040104310499e-02),
    (2.00000000000000e+00, 1.00000000000000e+00, 2.64241117657115e-01),
    (2.00000000000000e+00, 2.00000000000000e+00, 5.93994150290162e-01),
    (2.00000000000000e+00, 5.00000000000000e+00, 9.59572318005487e-01),
    (2.00000000000000e+00, 1.00000000000000e+01, 9.99500600772613e-01),
    (2.00000000000000e+00, 2.00000000000000e+01, 9.99999956715774e-01),
    (5.00000000000000e+00, 1.00000000000000e-01, 7.66780168618933e-08),
    (5.00000000000000e+00, 5.00000000000000e-01, 1.72115629955841e-04),
    (5.00000000000000e+00, 1.00000000000000e+00, 3.65984682734371e-03),
    (5.00000000000000e+00, 2.00000000000000e+00, 5.26530173437111e-02),
    (5.00000000000000e+00, 5.00000000000000e+00, 5.59506714934788e-01),
    (5.00000000000000e+00, 1.00000000000000e+01, 9.70747311923039e-01),
    (5.00000000000000e+00, 2.00000000000000e+01, 9.99983055256070e-01),
    (1.00000000000000e+01, 1.00000000000000e-01, 2.51634780677032e-17),
    (1.00000000000000e+01, 5.00000000000000e-01, 1.70967002934891e-10),
    (1.00000000000000e+01, 1.00000000000000e+00, 1.11425478338721e-07),
    (1.00000000000000e+01, 2.00000000000000e+00, 4.64980750172639e-05),
    (1.00000000000000e+01, 5.00000000000000e+00, 3.18280573062048e-02),
    (1.00000000000000e+01, 1.00000000000000e+01, 5.42070285528148e-01),
    (1.00000000000000e+01, 2.00000000000000e+01, 9.95004587691692e-01),
];

#[test]
fn test_gammp() {
    for &(a, x, expected) in GAMMP_TABLE.iter() {
        let result = gammp(a, x);
        let eps = f64::EPSILON + 1e-10 * result.abs();
        assert_abs_diff_eq!(result, expected, epsilon = eps);
    }
}

const GAMMQ_TABLE: [(f64, f64, f64); 42] = [
    (1.00000000000000e-01, 1.00000000000000e-01, 1.72448240414149e-01),
    (1.00000000000000e-01, 5.00000000000000e-01, 5.85975541098665e-02),
    (1.00000000000000e-01, 1.00000000000000e+00, 2.41273437263278e-02),
    (1.00000000000000e-01, 2.00000000000000e+00, 5.67382397981153e-03),
    (1.00000000000000e-01, 5.00000000000000e+00, 1.43938965846734e-04),
    (1.00000000000000e-01, 1.00000000000000e+01, 5.54798571790191e-07),
    (1.00000000000000e-01, 2.00000000000000e+01, 1.40135898021700e-11),
    (5.00000000000000e-01, 1.00000000000000e-01, 6.54720846018577e-01),
    (5.00000000000000e-01, 5.00000000000000e-01, 3.17310507862911e-01),
    (5.00000000000000e-01, 1.00000000000000e+00, 1.57299207050281e-01),
    (5.00000000000000e-01, 2.00000000000000e+00, 4.55002638963586e-02),
    (5.00000000000000e-01, 5.00000000000000e+00, 1.56540225800255e-03),
    (5.00000000000000e-01, 1.00000000000000e+01, 7.74421643104409e-06),
    (5.00000000000000e-01, 2.00000000000000e+01, 2.53962858947086e-10),
    (1.00000000000000e+00, 1.00000000000000e-01, 9.04837418035960e-01),
    (1.00000000000000e+00, 5.00000000000000e-01, 6.06530659712633e-01),
    (1.00000000000000e+00, 1.00000000000000e+00, 3.67879441171442e-01),
    (1.00000000000000e+00, 2.00000000000000e+00, 1.35335283236613e-01),
    (1.00000000000000e+00, 5.00000000000000e+00, 6.73794699908547e-03),
    (1.00000000000000e+00, 1.00000000000000e+01, 4.53999297624849e-05),
    (1.00000000000000e+00, 2.00000000000000e+01, 2.06115362243856e-09),
    (2.00000000000000e+00, 1.00000000000000e-01, 9.95321159839555e-01),
    (2.00000000000000e+00, 5.00000000000000e-01, 9.09795989568950e-01),
    (2.00000000000000e+00, 1.00000000000000e+00, 7.35758882342885e-01),
    (2.00000000000000e+00, 2.00000000000000e+00, 4.06005849709838e-01),
    (2.00000000000000e+00, 5.00000000000000e+00, 4.04276819945128e-02),
    (2.00000000000000e+00, 1.00000000000000e+01, 4.99399227387334e-04),
    (2.00000000000000e+00, 2.00000000000000e+01, 4.32842260712097e-08),
    (5.00000000000000e+00, 1.00000000000000e-01, 9.99999923321983e-01),
    (5.00000000000000e+00, 5.00000000000000e-01, 9.99827884370044e-01),
    (5.00000000000000e+00, 1.00000000000000e+00, 9.96340153172656e-01),
    (5.00000000000000e+00, 2.00000000000000e+00, 9.47346982656289e-01),
    (5.00000000000000e+00, 5.00000000000000e+00, 4.40493285065213e-01),
    (5.00000000000000e+00, 1.00000000000000e+01, 2.92526880769611e-02),
    (5.00000000000000e+00, 2.00000000000000e+01, 1.69447439300674e-05),
    (1.00000000000000e+01, 1.00000000000000e-01, 1.00000000000000e+00),
    (1.00000000000000e+01, 5.00000000000000e-01, 9.99999999829033e-01),
    (1.00000000000000e+01, 1.00000000000000e+00, 9.99999888574522e-01),
    (1.00000000000000e+01, 2.00000000000000e+00, 9.99953501924983e-01),
    (1.00000000000000e+01, 5.00000000000000e+00, 9.68171942693795e-01),
    (1.00000000000000e+01, 1.00000000000000e+01, 4.57929714471852e-01),
    (1.00000000000000e+01, 2.00000000000000e+01, 4.99541230830758e-03),
];

#[test]
fn test_gammq() {
    for &(a, x, expected) in GAMMQ_TABLE.iter() {
        let result = gammq(a, x);
        let epsilon = f64::EPSILON + 1e-9 * result.abs();
        if (result - expected).abs() > epsilon {
            dbg!(a, x, result, expected);
        }
        assert_abs_diff_eq!(result, expected, epsilon = epsilon);
    }
}

const INVGAMMP_TABLE: [(f64, f64, f64); 42] = [
    (1.00000000000000e-01, 1.00000000000000e-02, 6.07304836240791e-21),
    (1.00000000000000e-01, 1.00000000000000e-01, 6.07304836274321e-11),
    (1.00000000000000e-01, 2.50000000000000e-01, 5.79171329496962e-07),
    (1.00000000000000e-01, 5.00000000000000e-01, 5.93391104460228e-04),
    (1.00000000000000e-01, 7.50000000000000e-01, 3.53063580735584e-02),
    (1.00000000000000e-01, 9.00000000000000e-01, 2.66154553738837e-01),
    (1.00000000000000e-01, 9.90000000000000e-01, 1.58847781792950e+00),
    (5.00000000000000e-01, 1.00000000000000e-02, 7.85439289548509e-05),
    (5.00000000000000e-01, 1.00000000000000e-01, 7.89538704671561e-03),
    (5.00000000000000e-01, 2.50000000000000e-01, 5.07655221338108e-02),
    (5.00000000000000e-01, 5.00000000000000e-01, 2.27468211559786e-01),
    (5.00000000000000e-01, 7.50000000000000e-01, 6.61651848465733e-01),
    (5.00000000000000e-01, 9.00000000000000e-01, 1.35277172704770e+00),
    (5.00000000000000e-01, 9.90000000000000e-01, 3.31744830051061e+00),
    (1.00000000000000e+00, 1.00000000000000e-02, 1.00503358535014e-02),
    (1.00000000000000e+00, 1.00000000000000e-01, 1.05360515657826e-01),
    (1.00000000000000e+00, 2.50000000000000e-01, 2.87682072451781e-01),
    (1.00000000000000e+00, 5.00000000000000e-01, 6.93147180559946e-01),
    (1.00000000000000e+00, 7.50000000000000e-01, 1.38629436111989e+00),
    (1.00000000000000e+00, 9.00000000000000e-01, 2.30258509299405e+00),
    (1.00000000000000e+00, 9.90000000000000e-01, 4.60517018598809e+00),
    (2.00000000000000e+00, 1.00000000000000e-02, 1.48554740253266e-01),
    (2.00000000000000e+00, 1.00000000000000e-01, 5.31811608389612e-01),
    (2.00000000000000e+00, 2.50000000000000e-01, 9.61278763114777e-01),
    (2.00000000000000e+00, 5.00000000000000e-01, 1.67834699001666e+00),
    (2.00000000000000e+00, 7.50000000000000e-01, 2.69263452888970e+00),
    (2.00000000000000e+00, 9.00000000000000e-01, 3.88972016986743e+00),
    (2.00000000000000e+00, 9.90000000000000e-01, 6.63835206799381e+00),
    (5.00000000000000e+00, 1.00000000000000e-02, 1.27910608009360e+00),
    (5.00000000000000e+00, 1.00000000000000e-01, 2.43259102596266e+00),
    (5.00000000000000e+00, 2.50000000000000e-01, 3.36860038597732e+00),
    (5.00000000000000e+00, 5.00000000000000e-01, 4.67090888279598e+00),
    (5.00000000000000e+00, 7.50000000000000e-01, 6.27443069844469e+00),
    (5.00000000000000e+00, 9.00000000000000e-01, 7.99358958605263e+00),
    (5.00000000000000e+00, 9.90000000000000e-01, 1.16046255794772e+01),
    (1.00000000000000e+01, 1.00000000000000e-02, 4.13019916627320e+00),
    (1.00000000000000e+01, 1.00000000000000e-01, 6.22130460522503e+00),
    (1.00000000000000e+01, 2.50000000000000e-01, 7.72588676952386e+00),
    (1.00000000000000e+01, 5.00000000000000e-01, 9.66871461471413e+00),
    (1.00000000000000e+01, 7.50000000000000e-01, 1.19138460215154e+01),
    (1.00000000000000e+01, 9.00000000000000e-01, 1.42059902921528e+01),
    (1.00000000000000e+01, 9.90000000000000e-01, 1.87831173933125e+01),
];

#[test]
fn test_invgammp() {
    for &(a, p, expected) in INVGAMMP_TABLE.iter() {
        let result = invgammp(p, a);
        let epsilon = f64::EPSILON + 1e-9 * result.abs();
        assert_abs_diff_eq!(result, expected, epsilon = epsilon);
    }
}
