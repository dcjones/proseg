use numeric_literals::replace_float_literals;
use rand::Rng;
use rand_distr::{Distribution, Exp1};

use super::float::Float;

/*
 * Compute the (normalized) upper incomplete gamma function for the pair (p, x).
 *
 * We use the algorithm described in [1]. We use two continued fractions to
 * evaluate the function in the regions {0 < x <= p} and {0 <= p < x}
 * (algorithm 3 of [1]).
 *
 * We also use a terminating series to evaluate the normalized version for
 * integer and half-integer values of p <= 30 as described in [2]. This is
 * faster than the algorithm of [1] when p is small since not more than p terms
 * are required to evaluate the function.
 *
 * Parameters
 * ----------
 *  normalized : if true, the normalized upper incomplete gamma is returned,
 *      else the non-normalized version is returned for the arguments (p, x).
 *
 * References
 * ----------
 *  [1] Algorithm 1006: Fast and accurate evaluation of a generalized
 *      incomplete gamma function, Rémy Abergel and Lionel Moisan, ACM
 *      Transactions on Mathematical Software (TOMS), 2020. DOI: 10.1145/3365983
 *  [2] https://www.boost.org/doc/libs/1_71_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
 */
#[replace_float_literals(T::from(literal).unwrap())]
pub fn upper_incomplete_gamma<T: Float>(p: T, x: T, normalized: bool) -> T {
    if normalized {
        let p_int = p.floor();
        if p == p_int && p < 30.0 {
            let mut k = 1.0;
            let mut r = 1.0;
            let mut sum = 1.0;
            while k < p_int {
                r *= x / k;
                sum += r;
                k += 1.0;
            }
            return (-x).exp() * sum;
        } else if p == p_int + 0.5 && p < 30.0 {
            let mut k = 1.0;
            let one_sqrtpi = 0.5641895835477563;
            let sqrt_x = x.sqrt();
            let mut r = 1.0;
            let mut sum = 0.0;
            while k < p_int + 1.0 {
                r *= x / (k - 0.5);
                sum += r;
                k += 1.0;
            }

            return sqrt_x.erfc() + (-x).exp() * one_sqrtpi * sum / sqrt_x;
        }
    }

    let x_smaller = p >= x;
    let f = if x_smaller {
        confluent_x_smaller(p, x)
    } else {
        confluent_p_smaller(p, x)
    };

    let pgm_max_exp_t = T::from(PGM_MAX_EXP).unwrap();

    if normalized {
        let out = f * (-x + p * x.ln() - pgm_lgamma(p)).exp();
        if x_smaller {
            1.0 - out
        } else {
            out
        }
    } else if x_smaller {
        let lgam = pgm_lgamma(p);
        let exp_lgam = if lgam >= pgm_max_exp_t {
            pgm_max_exp_t.exp()
        } else {
            lgam.exp()
        };
        let arg = (-x + p * x.ln() - lgam)
            .min(pgm_max_exp_t)
            .max(-pgm_max_exp_t);

        (1.0 - f * arg.exp()) * exp_lgam
    } else {
        let arg = (-x + p * x.ln()).min(T::from(PGM_MAX_EXP).unwrap());
        f * arg.exp()
    }
}

const FLT_MIN: f64 = 1.17549e-38;
const FLT_EPSILON: f64 = 1.19209e-07;
const PGM_MAX_EXP: f64 = 88.7228;

/*
 * Compute function G(p, x) (A confluent hypergeometric function ratio).
 * This function is defined in equation 14 of [1] and this implementation
 * uses a continued fraction (eq. 15) defined for x <= p. The continued
 * fraction is evaluated using the Modified Lentz method.
 *
 * G(p, x) = a_1/b_1+ a_2/b_2+ a_3/b_3+ ..., such that a_1 = 1 and for n >= 1:
 * a_2n = -(p - 1 + n)*x, a_(2n+1) = n*x, b_n = p - 1 + n.
 *
 * Note that b_n can be reduced to b_1 = p, b_n = b_(n-1) + 1 for n >= 2. Also
 * for odd n, the argument of a_n is "k=(n-1)/2" and for even n "k=n/2". This
 * means we can pre-compute constant terms s = 0.5 * x and r = -(p - 1) * x.
 * This simplifies a_n into: a_n = s * (n - 1) for odd n and a_n = r - s * n
 * for even n >= 2. The terms for the first iteration are pre-calculated as
 * explained in [1].
 *
 * References
 * ----------
 *  [1] Algorithm 1006: Fast and accurate evaluation of a generalized
 *      incomplete gamma function, Rémy Abergel and Lionel Moisan, ACM
 *      Transactions on Mathematical Software (TOMS), 2020. DOI: 10.1145/3365983
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn confluent_x_smaller<T: Float>(p: T, x: T) -> T {
    let flt_min_t = T::from(FLT_MIN).unwrap();
    let mut a = 1.0;
    let mut b = p;
    let r = -(p - 1.0) * x;
    let s = 0.5 * x;
    let mut f = a / b;
    let mut c = a / flt_min_t;
    let mut d = 1.0 / b;

    for n in 2..100 {
        a = if n & 1 != 0 {
            s * T::from(n - 1).unwrap()
        } else {
            r - s * T::from(n).unwrap()
        };

        b += 1.0;
        c = b + a / c;
        c = c.max(flt_min_t);

        d = a * d + b;
        d = d.max(flt_min_t);

        d = d.recip();
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < T::from(FLT_EPSILON).unwrap() {
            break;
        }
    }

    f
}

/*
 * Compute function G(p, x) (A confluent hypergeometric function ratio).
 * This function is defined in equation 14 of [1] and this implementation
 * uses a continued fraction (eq. 16) defined for x > p. The continued
 * fraction is evaluated using the Modified Lentz method.
 *
 * G(p, x) = a_1/b_1+ a_2/b_2+ a_3/b_3+ ..., such that a_1 = 1 and for n > 1:
 * a_n = -(n - 1) * (n - p - 1), and for n >= 1: b_n = x + 2n - 1 - p.
 *
 * Note that b_n can be re-written as b_1 = x - p + 1 and
 * b_n = (((x - p + 1) + 2) + 2) + 2 ...) for n >= 2. Thus b_n = b_(n-1) + 2
 * for n >= 2. Also a_n can be re-written as a_n = (n - 1) * ((p - (n - 1)).
 * So if we can initialize the series with a_1 = 1 and instead of computing
 * (n - 1) at every iteration we can instead start the counter at n = 1 and
 * just compute a_(n+1) = n * (p - n). This doesnt affect b_n terms since all
 * we need is to keep incrementing b_n by 2 every iteration after initializing
 * the series with b_1 = x - p + 1.
 *
 * References
 * ----------
 *  [1] Algorithm 1006: Fast and accurate evaluation of a generalized
 *      incomplete gamma function, Rémy Abergel and Lionel Moisan, ACM
 *      Transactions on Mathematical Software (TOMS), 2020. DOI: 10.1145/3365983
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn confluent_p_smaller<T: Float>(p: T, x: T) -> T {
    let flt_min_t = T::from(FLT_MIN).unwrap();
    let mut a = 1.0;
    let mut b = x - p + 1.0;
    let mut f = a / b;
    let mut c = a / flt_min_t;
    let mut d = 1.0 / b;

    let mut n = 1.0;
    while n < 100.0 {
        a = n * (p - n);
        b += 2.0;

        c = b + a / c;
        c = c.max(flt_min_t);

        d = a * d + b;
        d = d.max(flt_min_t);

        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < T::from(FLT_EPSILON).unwrap() {
            break;
        }

        n += 1.0;
    }

    f
}

#[allow(clippy::excessive_precision)]
const LOG_FACTORIAL: [f64; 200] = [
    0.00000000000000000000,
    0.00000000000000000000,
    0.69314718055994530943,
    1.79175946922805500079,
    3.17805383034794561975,
    4.78749174278204599415,
    6.57925121201010099526,
    8.52516136106541430086,
    10.60460290274525022719,
    12.80182748008146961186,
    15.10441257307551529612,
    17.50230784587388584150,
    19.98721449566188614923,
    22.55216385312342288610,
    25.19122118273868150135,
    27.89927138384089156699,
    30.67186010608067280557,
    33.50507345013688888583,
    36.39544520803305357320,
    39.33988418719949403668,
    42.33561646075348502624,
    45.38013889847690802634,
    48.47118135183522388137,
    51.60667556776437357377,
    54.78472939811231919027,
    58.00360522298051993775,
    61.26170176100200198341,
    64.55753862700633105565,
    67.88974313718153497793,
    71.25703896716800901656,
    74.65823634883016438751,
    78.09222355331531063849,
    81.55795945611503718065,
    85.05446701758151741707,
    88.58082754219767880610,
    92.13617560368709247937,
    95.71969454214320249114,
    99.33061245478742692927,
    102.96819861451381269979,
    106.63176026064345913030,
    110.32063971475739543732,
    114.03421178146170323481,
    117.77188139974507154195,
    121.53308151543863396132,
    125.31727114935689513381,
    129.12393363912721488962,
    132.95257503561630989253,
    136.80272263732636846278,
    140.67392364823425940368,
    144.56574394634488600619,
    148.47776695177303207807,
    152.40959258449735784502,
    156.36083630307878519772,
    160.33112821663090702407,
    164.32011226319518140682,
    168.32744544842765233028,
    172.35279713916280155961,
    176.39584840699735171499,
    180.45629141754377104678,
    184.53382886144949050211,
    188.62817342367159119398,
    192.73904728784490243687,
    196.86618167288999400877,
    201.00931639928152667995,
    205.16819948264119853609,
    209.34258675253683563977,
    213.53224149456326118324,
    217.73693411395422725452,
    221.95644181913033395059,
    226.19054832372759332448,
    230.43904356577695233255,
    234.70172344281826774803,
    238.97838956183432307379,
    243.26884900298271419139,
    247.57291409618688395045,
    251.89040220972319437942,
    256.22113555000952545004,
    260.56494097186320932358,
    264.92164979855280104726,
    269.29109765101982254532,
    273.67312428569370413856,
    278.06757344036614290617,
    282.47429268763039605927,
    286.89313329542699396169,
    291.32395009427030757587,
    295.76660135076062402293,
    300.22094864701413177710,
    304.68685676566871547988,
    309.16419358014692195247,
    313.65282994987906178830,
    318.15263962020932683727,
    322.66349912672617686327,
    327.18528770377521719404,
    331.71788719692847316467,
    336.26118197919847702115,
    340.81505887079901787051,
    345.37940706226685413927,
    349.95411804077023693038,
    354.53908551944080887464,
    359.13420536957539877521,
    363.73937555556349016106,
    368.35449607240474959036,
    372.97946888568902071293,
    377.61419787391865648951,
    382.25858877306002911456,
    386.91254912321755249360,
    391.57598821732961960618,
    396.24881705179152582841,
    400.93094827891574549739,
    405.62229616114488922607,
    410.32277652693730540800,
    415.03230672824963956580,
    419.75080559954473413686,
    424.47819341825707464833,
    429.21439186665157011769,
    433.95932399501482021331,
    438.71291418612118484521,
    443.47508812091894095375,
    448.24577274538460572306,
    453.02489623849613509243,
    457.81238798127818109829,
    462.60817852687492218733,
    467.41219957160817877195,
    472.22438392698059622665,
    477.04466549258563309865,
    481.87297922988793424937,
    486.70926113683941224841,
    491.55344822329800347216,
    496.40547848721762064228,
    501.26529089157929280907,
    506.13282534203487522673,
    511.00802266523602676584,
    515.89082458782239759554,
    520.78117371604415142272,
    525.67901351599506276635,
    530.58428829443349222794,
    535.49694318016954425188,
    540.41692410599766910329,
    545.34417779115487379116,
    550.27865172428556556072,
    555.22029414689486986889,
    560.16905403727303813799,
    565.12488109487429888134,
    570.08772572513420617835,
    575.05753902471020677645,
    580.03427276713078114545,
    585.01787938883911766030,
    590.00831197561785385064,
    595.00552424938196893756,
    600.00947055532742813178,
    605.02010584942368387473,
    610.03738568623860821782,
    615.06126620708488456080,
    620.09170412847732001271,
    625.12865673089094925574,
    630.17208184781019580933,
    635.22193785505973290251,
    640.27818366040804093364,
    645.34077869343500771793,
    650.40968289565523929863,
    655.48485671088906617809,
    660.56626107587352919603,
    665.65385741110591327763,
    670.74760761191267560699,
    675.84747403973687401857,
    680.95341951363745458536,
    686.06540730199399785727,
    691.18340111441075296339,
    696.30736509381401183605,
    701.43726380873708536878,
    706.57306224578734715758,
    711.71472580229000698404,
    716.86222027910346005219,
    722.01551187360123895687,
    727.17456717281576800138,
    732.33935314673928201890,
    737.50983714177743377771,
    742.68598687435126293188,
    747.86777042464334813721,
    753.05515623048410311924,
    758.24811308137431348220,
    763.44661011264013927846,
    768.65061679971693459068,
    773.86010295255835550465,
    779.07503871016734109389,
    784.29539453524566594567,
    789.52114120895886717477,
    794.75224982581345378740,
    799.98869178864340312440,
    805.23043880370304542504,
    810.47746287586353153287,
    815.72973630391016147678,
    820.98723167593794297625,
    826.24992186484282852277,
    831.51778002390615662787,
    836.79077958246990348590,
    842.06889424170042068862,
    847.35209797043840918018,
    852.64036500113294436698,
    857.93366982585743685252,
];

#[replace_float_literals(T::from(literal).unwrap())]
pub fn pgm_lgamma<T: Float>(z: T) -> T {
    if z.floor() == z && z < 201.0 {
        T::from(LOG_FACTORIAL[(z - 1.0).as_usize()]).unwrap()
    } else {
        z.lgamma()
    }
}

#[replace_float_literals(T::from(literal).unwrap())]
pub fn random_left_bounded_gamma<R: Rng, T: Float>(rng: &mut R, a: T, b: T, t: T) -> T
where
    Exp1: Distribution<T>,
{
    if a > 1.0 {
        let b = t * b;
        let amin1 = a - 1.0;
        let bmina = b - a;
        let c0 = 0.5 * (bmina + ((bmina * bmina) + 4.0 * b).sqrt()) / b;
        let one_minus_c0 = 1.0 - c0;
        let log_m = amin1 * ((amin1 / one_minus_c0).ln() - 1.0);

        let mut x: T;
        // TODO: seems we sometimes get stuck here
        // b = 0
        // t = 0
        // c0 = 0
        loop {
            x = b + rng.sample::<T, Exp1>(Exp1) / c0;
            let threshold = amin1 * x.ln() - x * one_minus_c0 - log_m;

            if (-T::from(rng.random::<f32>()).unwrap()).ln_1p() <= threshold {
                break;
            }
        }
        t * (x / b)
    } else if a == 1.0 {
        t + rng.sample::<T, Exp1>(Exp1) / b
    } else {
        let amin1 = a - 1.0;
        let tb = t * b;
        let mut x;
        loop {
            x = 1.0 + rng.sample::<T, Exp1>(Exp1) / tb;
            if (-T::from(rng.random::<f32>()).unwrap()).ln_1p() <= amin1 * x.ln() {
                break;
            }
        }

        t * x
    }
}
