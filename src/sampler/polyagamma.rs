
// use num_traits::{cast::FromPrimitive, float::Float};
use numeric_literals::{replace_float_literals};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Normal, Exp1, Standard};

mod float;
use float::Float;

mod common;
mod saddlepoint;
use saddlepoint::sample_polyagamma_saddlepoint;


struct PolyaGamma<T: Float> {
    h: T,
    z: T,
}


#[replace_float_literals(T::from(literal).unwrap())]
fn sech<T: Float>(x: T) -> T {
    return 1.0 / x.cosh();
}


impl<T: Float> PolyaGamma<T>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    Exp1: Distribution<T>,
{
    fn new(h: T, z: T) -> Self {
        if h.is_sign_negative() {
            panic!("b must be non-negative")
        }

        return Self {
            h, z
        };
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn mean(&self) -> T {
        if self.z == T::zero() {
            return self.h / 4.0;
        } else {
            return self.h * 0.5 * self.z.recip() * (0.5 * self.z).tanh();
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn var(&self) -> T {
        if self.z == T::zero() {
            return self.h / 24.0;
        } else if self.z.sinh().is_infinite() {
            return self.h * 0.25 * (
                self.z.powi(3).recip() * 2.0*self.z.signum() -
                self.z.recip().powi(2) * sech(0.5*self.z).powi(2));
        } else {
            return self.h * 0.25 * self.z.powi(3).recip() *
                (self.z.sinh() - self.z) *
                sech(0.5*self.z).powi(2);
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
        if self.h >= 50.0 {
            return self.sample_normal(rng);
        } else {
            return self.sample_saddlepoint(rng);
        }
        // } else if self.h >= 8.0 || (self.h > 4.0 && self.z <= 4.0) {
        //     return self.sample_saddlepoint(rng);
        // } else if self.h == 1.0 || (self.h == self.h.floor() && self.z <= 1.0) {
        //     return self.sample_devroye(rng);
        // } else {
        //     return self.sample_alternate(rng);
        // }
    }

    fn sample_normal<R: Rng>(&self, rng: &mut R) -> T {
        return Normal::new(
            self.mean(),
            self.var().sqrt()
        ).unwrap().sample(rng);
    }

    fn sample_saddlepoint<R: Rng>(&self, rng: &mut R) -> T {
        return sample_polyagamma_saddlepoint(rng, self.h, self.z);
    }

}


