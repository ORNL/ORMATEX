/*
 * Copyright© 2025,2026 UT-Battelle, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/// Time step size controller.
/// Used with integrators that provided an embedded error estimate
/// such as exprb32.
///
use crate::ode_sys::StepResult;
use faer::prelude::*;

/// Interface for error-based time-step controllers.
pub trait TimeStepController {
    /// Decide whether a proposed step is accepted and return the next step size.
    fn control(&self, step: &StepResult<f64, Mat<f64>>, order: usize) -> (bool, f64);
}

/// Error-based time-step controller for steppers with an error estimate.
pub struct BasicStepController {
    atol: f64,
    rtol: f64,
    safety: f64,
    min_factor: f64,
    max_factor: f64,
    min_dt: f64,
    max_dt: f64,
}

impl BasicStepController {
    /// Create a controller with absolute and relative error tolerances.
    pub fn new(
        atol: f64,
        rtol: f64,
        safety: f64,
        min_factor: f64,
        max_factor: f64,
        min_dt: f64,
        max_dt: f64,
    ) -> Self {
        assert!(
            atol >= 0.0 && rtol >= 0.0 && (atol > 0.0 || rtol > 0.0),
            "atol and rtol must not both be zero"
        );
        assert!(
            safety > 0.0 && safety <= 1.0,
            "safety must be in (0, 1]"
        );
        assert!(
            min_factor > 0.0 && max_factor >= min_factor,
            "invalid step-size factor limits"
        );
        assert!(
            min_dt > 0.0 && max_dt >= min_dt,
            "invalid step-size limits"
        );

        Self {
            atol,
            rtol,
            safety,
            min_factor,
            max_factor,
            min_dt,
            max_dt,
        }
    }
}

impl Default for BasicStepController {
    fn default() -> Self {
        Self::new(1.0e-6, 1.0e-3, 0.9, 0.2, 5.0, 1.0e-14, f64::INFINITY)
    }
}

impl TimeStepController for BasicStepController {
    fn control(&self, step: &StepResult<f64, Mat<f64>>, order: usize) -> (bool, f64) {
        let Some(err) = step.err else {
            return (true, step.dt);
        };

        let tolerance = self.atol + self.rtol * step.y.as_ref().norm_max();
        let error_ratio = err / tolerance;
        let accepted = error_ratio <= 1.0;
        let factor = if error_ratio == 0.0 {
            self.max_factor
        } else {
            self.safety * error_ratio.powf(-1.0 / (order as f64 + 1.0))
        }
        .clamp(self.min_factor, self.max_factor);
        let next_dt = (step.dt.abs() * factor)
            .clamp(self.min_dt, self.max_dt)
            .copysign(step.dt);

        (accepted, next_dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(dt: f64, y: &[f64], err: Option<f64>) -> StepResult<f64, Mat<f64>> {
        StepResult::new(dt, dt, faer::Mat::from_fn(y.len(), 1, |i, _| y[i]), err)
    }

    #[test]
    fn accepts_and_grows_for_small_error() {
        let controller = BasicStepController::new(0.1, 0.0, 0.9, 0.2, 5.0, 1.0e-14, f64::INFINITY);

        let (accepted, next_dt) = controller.control(&step(0.1, &[1.0], Some(0.01)), 2);

        assert!(accepted);
        assert!(next_dt > 0.1);
        assert!(next_dt <= 0.5);
    }

    #[test]
    fn rejects_and_shrinks_for_large_error() {
        let controller = BasicStepController::new(0.1, 0.0, 0.9, 0.2, 5.0, 1.0e-14, f64::INFINITY);

        let (accepted, next_dt) = controller.control(&step(0.1, &[1.0], Some(1.0)), 2);

        assert!(!accepted);
        assert!(next_dt < 0.1);
    }

    #[test]
    fn accepts_steps_without_error_estimates() {
        let controller = BasicStepController::default();
        let (accepted, next_dt) = controller.control(&step(0.1, &[1.0], None), 2);

        assert!(accepted);
        assert_eq!(next_dt, 0.1);
    }

    #[test]
    fn uses_infinity_norm_and_clamps_step_size() {
        let controller = BasicStepController::new(0.0, 0.1, 0.9, 0.5, 2.0, 0.1, 0.3);

        let (accepted, next_dt) = controller.control(&step(0.01, &[2.0, 10.0], Some(0.9)), 2);
        assert!(accepted);
        assert_eq!(next_dt, 0.1);

        let (_, next_dt) = controller.control(&step(0.2, &[2.0, 10.0], Some(0.0)), 2);
        assert_eq!(next_dt, 0.3);
    }

    #[test]
    fn preserves_negative_step_sign() {
        let controller = BasicStepController::default();
        let (_, next_dt) = controller.control(&step(-0.1, &[1.0], Some(0.0)), 2);

        assert!(next_dt < 0.0);
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(std::panic::catch_unwind(|| {
            BasicStepController::new(0.0, 0.0, 0.9, 0.2, 5.0, 1.0e-14, f64::INFINITY);
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            BasicStepController::new(1.0, 0.0, 0.0, 0.2, 5.0, 1.0e-14, f64::INFINITY);
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            BasicStepController::new(1.0, 0.0, 0.9, 0.0, 5.0, 1.0e-14, f64::INFINITY);
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            BasicStepController::new(1.0, 0.0, 0.9, 0.2, 5.0, 0.0, f64::INFINITY);
        })
        .is_err());
    }
}
